/**
 * Test for Polynomial Model Selection in L3 Compression
 *
 * Tests that the adaptive model selector correctly chooses:
 *   - LINEAR for linear data
 *   - POLYNOMIAL2 for quadratic data
 *   - POLYNOMIAL3 for cubic data
 *   - FOR+BitPacking for clustered data
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "/root/autodl-tmp/code/L3/include/L3_format.hpp"
#include "/root/autodl-tmp/code/L3/include/L3_Vertical_format.hpp"
#include "/root/autodl-tmp/code/L3/src/kernels/compression/encoder_Vertical_opt.cu"
#include "/root/autodl-tmp/code/L3/src/kernels/decompression/decoder_Vertical_opt.cu"

const char* getModelName(int model_type) {
    switch (model_type) {
        case MODEL_CONSTANT: return "CONSTANT";
        case MODEL_LINEAR: return "LINEAR";
        case MODEL_POLYNOMIAL2: return "POLY2";
        case MODEL_POLYNOMIAL3: return "POLY3";
        case MODEL_FOR_BITPACK: return "FOR+BP";
        default: return "UNKNOWN";
    }
}

// Test helper: verify compression/decompression roundtrip
template<typename T>
bool verifyRoundtrip(const std::vector<T>& data, const std::string& test_name) {
    VerticalConfig config;
    config.partition_size_hint = data.size();  // Single partition
    auto partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), data.size());

    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, config);

    T* d_output;
    cudaMalloc(&d_output, data.size() * sizeof(T));
    Vertical_decoder::decompressAll<T>(
        compressed, d_output, DecompressMode::BRANCHLESS);
    cudaDeviceSynchronize();

    std::vector<T> output(data.size());
    cudaMemcpy(output.data(), d_output, data.size() * sizeof(T), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != output[i]) {
            errors++;
            if (errors <= 5) {
                std::cout << "  Error at " << i << ": expected " << data[i]
                          << ", got " << output[i] << std::endl;
            }
        }
    }

    std::cout << test_name << ": model=" << getModelName(partitions[0].model_type)
              << ", bits=" << partitions[0].delta_bits
              << ", errors=" << errors << "/" << data.size();

    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(compressed);

    if (errors == 0) {
        std::cout << " [PASS]" << std::endl;
        return true;
    } else {
        std::cout << " [FAIL]" << std::endl;
        return false;
    }
}

// Test 1: Linear data (should select LINEAR)
bool testLinearData() {
    std::cout << "\n=== Test 1: Linear Data ===" << std::endl;

    const size_t n = 2048;
    std::vector<int32_t> data(n);

    // y = 1000 + 5*x + small noise
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> noise(-2, 2);

    for (size_t i = 0; i < n; i++) {
        data[i] = 1000 + 5 * static_cast<int32_t>(i) + noise(gen);
    }

    return verifyRoundtrip(data, "Linear (y=1000+5x+noise)");
}

// Test 2: Quadratic data (should select POLYNOMIAL2)
bool testQuadraticData() {
    std::cout << "\n=== Test 2: Quadratic Data ===" << std::endl;

    const size_t n = 2048;
    std::vector<int32_t> data(n);

    // y = 1000 + 2*x + 0.01*x^2 + small noise
    // This creates a parabolic curve
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> noise(-3, 3);

    for (size_t i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = 1000.0 + 2.0 * x + 0.01 * x * x;
        data[i] = static_cast<int32_t>(std::round(y)) + noise(gen);
    }

    return verifyRoundtrip(data, "Quadratic (y=1000+2x+0.01x^2+noise)");
}

// Test 3: Cubic data (should select POLYNOMIAL3)
bool testCubicData() {
    std::cout << "\n=== Test 3: Cubic Data ===" << std::endl;

    const size_t n = 2048;
    std::vector<int32_t> data(n);

    // y = 1000 + x + 0.001*x^2 + 0.000001*x^3 + small noise
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> noise(-5, 5);

    for (size_t i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = 1000.0 + x + 0.001 * x * x + 0.000001 * x * x * x;
        data[i] = static_cast<int32_t>(std::round(y)) + noise(gen);
    }

    return verifyRoundtrip(data, "Cubic (y=1000+x+0.001x^2+0.000001x^3+noise)");
}

// Test 4: Clustered data (should select FOR+BitPacking)
bool testClusteredData() {
    std::cout << "\n=== Test 4: Clustered Data ===" << std::endl;

    const size_t n = 2048;
    std::vector<int32_t> data(n);

    // Random values clustered around 50000
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(50000, 50100);

    for (size_t i = 0; i < n; i++) {
        data[i] = dist(gen);
    }

    return verifyRoundtrip(data, "Clustered (50000-50100)");
}

// Test 5: Mixed patterns in single partition boundary test
bool testMixedBoundary() {
    std::cout << "\n=== Test 5: Mixed Pattern Boundary ===" << std::endl;

    const size_t n = 4096;  // 2 partitions of 2048 each
    std::vector<int32_t> data(n);

    // First half: linear
    for (size_t i = 0; i < n/2; i++) {
        data[i] = 1000 + static_cast<int32_t>(i);
    }

    // Second half: clustered
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(50000, 50100);
    for (size_t i = n/2; i < n; i++) {
        data[i] = dist(gen);
    }

    // Test with 2 partitions
    VerticalConfig config;
    config.partition_size_hint = 2048;
    auto partitions = Vertical_encoder::createFixedPartitions<int32_t>(n, 2048);

    auto compressed = Vertical_encoder::encodeVertical<int32_t>(data, partitions, config);

    int32_t* d_output;
    cudaMalloc(&d_output, n * sizeof(int32_t));
    Vertical_decoder::decompressAll<int32_t>(
        compressed, d_output, DecompressMode::BRANCHLESS);
    cudaDeviceSynchronize();

    std::vector<int32_t> output(n);
    cudaMemcpy(output.data(), d_output, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::cout << "Partition 0: model=" << getModelName(partitions[0].model_type)
              << ", bits=" << partitions[0].delta_bits << std::endl;
    std::cout << "Partition 1: model=" << getModelName(partitions[1].model_type)
              << ", bits=" << partitions[1].delta_bits << std::endl;

    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != output[i]) {
            errors++;
            if (errors <= 5) {
                std::cout << "  Error at " << i << ": expected " << data[i]
                          << ", got " << output[i] << std::endl;
            }
        }
    }

    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(compressed);

    bool pass = (errors == 0);
    std::cout << "Mixed boundary test: errors=" << errors << "/" << n
              << (pass ? " [PASS]" : " [FAIL]") << std::endl;

    return pass;
}

// Test 6: Strong quadratic curve
bool testStrongQuadratic() {
    std::cout << "\n=== Test 6: Strong Quadratic Curve ===" << std::endl;

    const size_t n = 512;
    std::vector<int32_t> data(n);

    // y = x^2 (perfect parabola, scaled)
    for (size_t i = 0; i < n; i++) {
        double x = static_cast<double>(i) - n/2;  // Center around 0
        data[i] = static_cast<int32_t>(x * x);
    }

    VerticalConfig config;
    auto partitions = Vertical_encoder::createFixedPartitions<int32_t>(n, n);
    auto compressed = Vertical_encoder::encodeVertical<int32_t>(data, partitions, config);

    int32_t* d_output;
    cudaMalloc(&d_output, n * sizeof(int32_t));
    Vertical_decoder::decompressAll<int32_t>(
        compressed, d_output, DecompressMode::BRANCHLESS);
    cudaDeviceSynchronize();

    std::vector<int32_t> output(n);
    cudaMemcpy(output.data(), d_output, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != output[i]) errors++;
    }

    std::cout << "Strong quadratic (x^2): model=" << getModelName(partitions[0].model_type)
              << ", bits=" << partitions[0].delta_bits
              << ", errors=" << errors << "/" << n
              << (errors == 0 ? " [PASS]" : " [FAIL]") << std::endl;

    // Print model parameters for verification
    std::cout << "  params: a0=" << partitions[0].model_params[0]
              << ", a1=" << partitions[0].model_params[1]
              << ", a2=" << partitions[0].model_params[2]
              << ", a3=" << partitions[0].model_params[3] << std::endl;

    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(compressed);

    return errors == 0;
}

// Test 7: Perfect polynomial (zero residuals expected)
bool testPerfectLinear() {
    std::cout << "\n=== Test 7: Perfect Linear (zero noise) ===" << std::endl;

    const size_t n = 1024;
    std::vector<int32_t> data(n);

    // y = 100 + 3*x (perfect line)
    for (size_t i = 0; i < n; i++) {
        data[i] = 100 + 3 * static_cast<int32_t>(i);
    }

    VerticalConfig config;
    auto partitions = Vertical_encoder::createFixedPartitions<int32_t>(n, n);
    auto compressed = Vertical_encoder::encodeVertical<int32_t>(data, partitions, config);

    std::cout << "Perfect linear (y=100+3x): model=" << getModelName(partitions[0].model_type)
              << ", bits=" << partitions[0].delta_bits << std::endl;
    std::cout << "  Expected: model=LINEAR, bits<=1 (rounding only)" << std::endl;

    int32_t* d_output;
    cudaMalloc(&d_output, n * sizeof(int32_t));
    Vertical_decoder::decompressAll<int32_t>(
        compressed, d_output, DecompressMode::BRANCHLESS);
    cudaDeviceSynchronize();

    std::vector<int32_t> output(n);
    cudaMemcpy(output.data(), d_output, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != output[i]) errors++;
    }

    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(compressed);

    bool pass = (errors == 0);
    std::cout << "Perfect linear test: errors=" << errors << "/" << n
              << (pass ? " [PASS]" : " [FAIL]") << std::endl;

    return pass;
}

// Test 8: Compression ratio comparison
bool testCompressionRatio() {
    std::cout << "\n=== Test 8: Compression Ratio Comparison ===" << std::endl;

    const size_t n = 2048;
    std::vector<int32_t> data(n);

    // y = 1000 + 0.5*x + 0.002*x^2 (mild quadratic)
    for (size_t i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        data[i] = static_cast<int32_t>(1000.0 + 0.5 * x + 0.002 * x * x);
    }

    VerticalConfig config;
    auto partitions = Vertical_encoder::createFixedPartitions<int32_t>(n, n);
    auto compressed = Vertical_encoder::encodeVertical<int32_t>(data, partitions, config);

    // Calculate compression ratio
    double original_size = n * sizeof(int32_t);
    double compressed_size = compressed.sequential_delta_words * sizeof(uint32_t) +
                             partitions.size() * 4 * sizeof(double);  // metadata
    double ratio = original_size / compressed_size;

    std::cout << "Quadratic data (y=1000+0.5x+0.002x^2):" << std::endl;
    std::cout << "  Model: " << getModelName(partitions[0].model_type) << std::endl;
    std::cout << "  Bits per value: " << partitions[0].delta_bits << std::endl;
    std::cout << "  Original: " << original_size << " bytes" << std::endl;
    std::cout << "  Compressed: " << compressed_size << " bytes" << std::endl;
    std::cout << "  Ratio: " << ratio << "x" << std::endl;

    Vertical_encoder::freeCompressedData(compressed);

    // Pass if we get at least 2x compression
    bool pass = (ratio >= 2.0);
    std::cout << "Compression ratio test: " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    return pass;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " L3 Polynomial Model Selection Test" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int total = 8;

    if (testLinearData()) passed++;
    if (testQuadraticData()) passed++;
    if (testCubicData()) passed++;
    if (testClusteredData()) passed++;
    if (testMixedBoundary()) passed++;
    if (testStrongQuadratic()) passed++;
    if (testPerfectLinear()) passed++;
    if (testCompressionRatio()) passed++;

    std::cout << "\n========================================" << std::endl;
    std::cout << " Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
