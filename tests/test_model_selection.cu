#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/kernels/compression/adaptive_selector.cuh"
#include "../include/L3_format.hpp"

int main() {
    // Load first partitions of normal dataset
    std::string filepath = "data/sosd/2-normal_200M_uint64.bin";
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file\n";
        return 1;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_elements = file_size / sizeof(uint64_t);
    std::vector<uint64_t> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    std::cout << "Loaded " << num_elements << " elements\n\n";
    
    // Test first 5 partitions
    int partition_size = 4096;
    for (int p = 0; p < 5; p++) {
        int start = p * partition_size;
        int end = std::min((p + 1) * partition_size, static_cast<int>(num_elements));
        
        std::cout << "Partition " << p << " [" << start << ", " << end << "]\n";
        
        // Test LINEAR
        auto decision_linear = adaptive_selector::computeFixedModelCPU<uint64_t>(
            data.data(), start, end, MODEL_LINEAR);
        std::cout << "  LINEAR: model_type=" << decision_linear.model_type 
                  << " delta_bits=" << decision_linear.delta_bits 
                  << " params=[" << decision_linear.params[0] << ", " << decision_linear.params[1] << "]\n";
        
        // Test ADAPTIVE
        auto decision_adaptive = adaptive_selector::computeDecisionCPU<uint64_t>(
            data.data(), start, end);
        std::cout << "  ADAPTIVE: model_type=" << decision_adaptive.model_type 
                  << " delta_bits=" << decision_adaptive.delta_bits << "\n";
        
        // Test FOR
        auto decision_for = adaptive_selector::computeFixedModelCPU<uint64_t>(
            data.data(), start, end, MODEL_FOR_BITPACK);
        std::cout << "  FOR: model_type=" << decision_for.model_type 
                  << " delta_bits=" << decision_for.delta_bits << "\n";
        
        std::cout << "\n";
    }
    
    return 0;
}
