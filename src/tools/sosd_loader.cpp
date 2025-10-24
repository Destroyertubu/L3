#include "sosd_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

// Helper: check if file ends with suffix
static bool endsWith(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Extract dataset name from filepath
std::string SOSDLoader::getDatasetName(const std::string& filepath) {
    size_t last_slash = filepath.find_last_of("/\\");
    size_t last_dot = filepath.find_last_of('.');

    if (last_slash == std::string::npos) last_slash = 0;
    else last_slash++;

    if (last_dot == std::string::npos || last_dot < last_slash) {
        return filepath.substr(last_slash);
    }

    return filepath.substr(last_slash, last_dot - last_slash);
}

// Check if file is binary format
bool SOSDLoader::isBinaryFormat(const std::string& filepath) {
    return endsWith(filepath, ".bin");
}

// Get dataset info
bool SOSDLoader::getDatasetInfo(const std::string& filepath,
                                size_t& num_elements,
                                size_t& file_size,
                                std::string& data_type) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return false;
    }

    file_size = file.tellg();
    file.seekg(0);

    // Determine data type from filename
    if (filepath.find("uint64") != std::string::npos) {
        data_type = "uint64";
    } else {
        data_type = "uint32";
    }

    if (isBinaryFormat(filepath)) {
        // Binary format: 8-byte header + data
        uint64_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
        num_elements = count;
    } else {
        // Text format: count lines
        std::string line;
        num_elements = 0;
        while (std::getline(file, line)) {
            if (!line.empty()) num_elements++;
        }
    }

    file.close();
    return true;
}

// Load binary format
template<typename T>
bool SOSDLoader::loadBinary(const std::string& filepath, std::vector<T>& data) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open binary file: " << filepath << std::endl;
        return false;
    }

    // Read 8-byte header (element count)
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));

    if (count == 0 || count > 1000000000ULL) {  // Sanity check
        std::cerr << "Invalid element count: " << count << std::endl;
        return false;
    }

    // Allocate and read data
    data.resize(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));

    if (!file.good() && !file.eof()) {
        std::cerr << "Error reading binary data" << std::endl;
        return false;
    }

    file.close();
    std::cout << "Loaded " << data.size() << " elements from " << filepath << std::endl;
    return true;
}

// Load text format
template<typename T>
bool SOSDLoader::loadText(const std::string& filepath, std::vector<T>& data) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open text file: " << filepath << std::endl;
        return false;
    }

    data.clear();
    data.reserve(20000000);  // Pre-allocate for typical size

    std::string line;
    uint64_t value;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        if (iss >> value) {
            data.push_back(static_cast<T>(value));
        }
    }

    file.close();
    std::cout << "Loaded " << data.size() << " elements from " << filepath << std::endl;
    return true;
}

// Auto-detect and load dataset
template<typename T>
bool SOSDLoader::loadDataset(const std::string& filepath, std::vector<T>& data) {
    if (isBinaryFormat(filepath)) {
        return loadBinary<T>(filepath, data);
    } else {
        return loadText<T>(filepath, data);
    }
}

// Explicit template instantiations
template bool SOSDLoader::loadDataset<uint32_t>(const std::string&, std::vector<uint32_t>&);
template bool SOSDLoader::loadDataset<uint64_t>(const std::string&, std::vector<uint64_t>&);
template bool SOSDLoader::loadDataset<int32_t>(const std::string&, std::vector<int32_t>&);
template bool SOSDLoader::loadDataset<int64_t>(const std::string&, std::vector<int64_t>&);

template bool SOSDLoader::loadBinary<uint32_t>(const std::string&, std::vector<uint32_t>&);
template bool SOSDLoader::loadBinary<uint64_t>(const std::string&, std::vector<uint64_t>&);
template bool SOSDLoader::loadBinary<int32_t>(const std::string&, std::vector<int32_t>&);
template bool SOSDLoader::loadBinary<int64_t>(const std::string&, std::vector<int64_t>&);

template bool SOSDLoader::loadText<uint32_t>(const std::string&, std::vector<uint32_t>&);
template bool SOSDLoader::loadText<uint64_t>(const std::string&, std::vector<uint64_t>&);
template bool SOSDLoader::loadText<int32_t>(const std::string&, std::vector<int32_t>&);
template bool SOSDLoader::loadText<int64_t>(const std::string&, std::vector<int64_t>&);
