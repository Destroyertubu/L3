/**
 * Debug: Why LINEAR model is not being selected for linear data
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

#include "L3_format.hpp"
#include "encoder_cost_optimal_gpu_merge.cuh"

// Load raw binary dataset
std::vector<uint64_t> loadRawDataset(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t count = file_size / sizeof(uint64_t);
    if (max_elements > 0 && count > max_elements) {
        count = max_elements;
    }

    std::vector<uint64_t> data(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint64_t));

    return data;
}

void debugPartition(const std::vector<uint64_t>& data, int start, int end) {
    int n = end - start;
    std::cout << "\n=== Debug Partition [" << start << ", " << end << ") ===" << std::endl;
    std::cout << "Size: " << n << " elements" << std::endl;

    // Show first few values
    std::cout << "\nFirst 10 values:" << std::endl;
    for (int i = 0; i < std::min(10, n); i++) {
        std::cout << "  data[" << (start + i) << "] = " << data[start + i] << std::endl;
    }

    // Calculate linear fit
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < n; i++) {
        double x = i;
        double y = static_cast<double>(data[start + i]);
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    double dn = static_cast<double>(n);
    double det = dn * sum_x2 - sum_x * sum_x;
    double theta0_linear = (sum_x2 * sum_y - sum_x * sum_xy) / det;
    double theta1_linear = (dn * sum_xy - sum_x * sum_y) / det;

    std::cout << "\nLINEAR fit:" << std::endl;
    std::cout << "  theta0 = " << std::scientific << std::setprecision(10) << theta0_linear << std::endl;
    std::cout << "  theta1 = " << theta1_linear << std::endl;

    // Calculate POLY2 fit
    double sum_x3 = 0, sum_x4 = 0, sum_x2y = 0;
    for (int i = 0; i < n; i++) {
        double x = i;
        double y = static_cast<double>(data[start + i]);
        double x2 = x * x;
        sum_x3 += x2 * x;
        sum_x4 += x2 * x2;
        sum_x2y += x2 * y;
    }

    double a00 = dn, a01 = sum_x, a02 = sum_x2;
    double a10 = sum_x, a11 = sum_x2, a12 = sum_x3;
    double a20 = sum_x2, a21 = sum_x3, a22 = sum_x4;
    double b0 = sum_y, b1 = sum_xy, b2 = sum_x2y;

    double det2 = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
    double theta0_poly2 = (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / det2;
    double theta1_poly2 = (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / det2;
    double theta2_poly2 = (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / det2;

    std::cout << "\nPOLY2 fit:" << std::endl;
    std::cout << "  theta0 = " << theta0_poly2 << std::endl;
    std::cout << "  theta1 = " << theta1_poly2 << std::endl;
    std::cout << "  theta2 = " << theta2_poly2 << std::endl;

    // Calculate max errors for both models
    long long linear_max_err = 0, poly2_max_err = 0;
    for (int i = 0; i < n; i++) {
        double x = i;
        uint64_t val = data[start + i];

        // LINEAR
        double pred_linear = theta0_linear + theta1_linear * x;
        int64_t pv_linear = static_cast<int64_t>(std::llrint(pred_linear));
        int64_t err_linear = static_cast<int64_t>(val) - pv_linear;
        linear_max_err = std::max(linear_max_err, (long long)std::abs(err_linear));

        // POLY2
        double pred_poly2 = theta0_poly2 + x * (theta1_poly2 + x * theta2_poly2);
        int64_t pv_poly2 = static_cast<int64_t>(std::llrint(pred_poly2));
        int64_t err_poly2 = static_cast<int64_t>(val) - pv_poly2;
        poly2_max_err = std::max(poly2_max_err, (long long)std::abs(err_poly2));
    }

    int linear_bits = (linear_max_err > 0) ? (64 - __builtin_clzll(linear_max_err)) + 1 : 0;
    int poly2_bits = (poly2_max_err > 0) ? (64 - __builtin_clzll(poly2_max_err)) + 1 : 0;

    std::cout << "\nMax errors:" << std::endl;
    std::cout << "  LINEAR max_error = " << linear_max_err << " -> " << linear_bits << " bits" << std::endl;
    std::cout << "  POLY2 max_error = " << poly2_max_err << " -> " << poly2_bits << " bits" << std::endl;

    // Calculate costs
    float linear_cost = 16.0f + n * linear_bits / 8.0f;
    float poly2_cost = 24.0f + n * poly2_bits / 8.0f;

    std::cout << "\nCosts:" << std::endl;
    std::cout << "  LINEAR cost = " << linear_cost << " bytes" << std::endl;
    std::cout << "  POLY2 cost = " << poly2_cost << " bytes" << std::endl;
    std::cout << "  LINEAR * 0.95 = " << (linear_cost * 0.95f) << std::endl;
    std::cout << "  POLY2 < LINEAR*0.95? " << (poly2_cost < linear_cost * 0.95f ? "YES" : "NO") << std::endl;

    // Show sample predictions
    std::cout << "\nSample predictions (first 5):" << std::endl;
    for (int i = 0; i < std::min(5, n); i++) {
        double x = i;
        uint64_t val = data[start + i];
        double pred_linear = theta0_linear + theta1_linear * x;
        double pred_poly2 = theta0_poly2 + x * (theta1_poly2 + x * theta2_poly2);
        std::cout << "  i=" << i << ": val=" << val
                  << ", linear_pred=" << std::llrint(pred_linear)
                  << ", poly2_pred=" << std::llrint(pred_poly2)
                  << ", linear_err=" << (static_cast<int64_t>(val) - std::llrint(pred_linear))
                  << ", poly2_err=" << (static_cast<int64_t>(val) - std::llrint(pred_poly2))
                  << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string filename = "data/sosd/1-linear_200M_uint64.bin";
    size_t max_elements = 10000000;  // 10M like original test

    std::cout << "Debug: Linear Model Selection Analysis" << std::endl;
    std::cout << "=======================================" << std::endl;

    auto data = loadRawDataset(filename, max_elements);
    if (data.empty()) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;

    // Show overall data characteristics
    std::cout << "\n=== Data Characteristics ===" << std::endl;
    std::cout << "First 20 values:" << std::endl;
    for (int i = 0; i < std::min(20, (int)data.size()); i++) {
        std::cout << "  data[" << i << "] = " << data[i] << std::endl;
    }

    // Check if data is truly linear
    std::cout << "\n=== Checking linearity ===" << std::endl;
    std::cout << "Differences (data[i+1] - data[i]):" << std::endl;
    for (int i = 0; i < std::min(19, (int)data.size()-1); i++) {
        int64_t diff = static_cast<int64_t>(data[i+1]) - static_cast<int64_t>(data[i]);
        std::cout << "  diff[" << i << "] = " << diff << std::endl;
    }

    // Debug first few partitions of different sizes
    debugPartition(data, 0, 2048);
    debugPartition(data, 2048, 4096);
    debugPartition(data, 4096, 6144);

    // Now run the actual partitioner and see what it produces
    std::cout << "\n\n=== Running Partitioner ===" << std::endl;

    CostOptimalConfig config = CostOptimalConfig::polynomialEnabled();
    config.min_partition_size = 256;
    config.max_partition_size = 8192;
    config.target_partition_size = 2048;

    GPUCostOptimalPartitioner<uint64_t> partitioner(data, config);
    auto partitions = partitioner.partition();

    std::cout << "Total partitions: " << partitions.size() << std::endl;

    // Count models
    int linear_count = 0, poly2_count = 0, poly3_count = 0, for_count = 0;
    for (const auto& p : partitions) {
        if (p.model_type == MODEL_LINEAR) linear_count++;
        else if (p.model_type == MODEL_POLYNOMIAL2) poly2_count++;
        else if (p.model_type == MODEL_POLYNOMIAL3) poly3_count++;
        else if (p.model_type == MODEL_FOR_BITPACK) for_count++;
    }

    std::cout << "LINEAR: " << linear_count << std::endl;
    std::cout << "POLY2: " << poly2_count << std::endl;
    std::cout << "POLY3: " << poly3_count << std::endl;
    std::cout << "FOR_BITPACK: " << for_count << std::endl;

    // Show first partition from partitioner
    if (!partitions.empty()) {
        const auto& p = partitions[0];
        std::cout << "\nFirst partition from partitioner:" << std::endl;
        std::cout << "  Range: [" << p.start_idx << ", " << p.end_idx << ")" << std::endl;
        std::cout << "  Model: " << (p.model_type == MODEL_LINEAR ? "LINEAR" :
                                      p.model_type == MODEL_POLYNOMIAL2 ? "POLY2" :
                                      p.model_type == MODEL_POLYNOMIAL3 ? "POLY3" : "FOR_BITPACK") << std::endl;
        std::cout << "  theta0: " << std::scientific << p.model_params[0] << std::endl;
        std::cout << "  theta1: " << p.model_params[1] << std::endl;
        std::cout << "  theta2: " << p.model_params[2] << std::endl;
        std::cout << "  delta_bits: " << p.delta_bits << std::endl;
    }

    return 0;
}
