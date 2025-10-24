#ifndef SOSD_LOADER_H
#define SOSD_LOADER_H

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

// SOSD dataset loader for both binary and text formats
class SOSDLoader {
public:
    // Load dataset from file (auto-detects format)
    template<typename T>
    static bool loadDataset(const std::string& filepath, std::vector<T>& data);

    // Load binary format (.bin files)
    template<typename T>
    static bool loadBinary(const std::string& filepath, std::vector<T>& data);

    // Load text format (.txt files)
    template<typename T>
    static bool loadText(const std::string& filepath, std::vector<T>& data);

    // Get dataset info without loading
    static bool getDatasetInfo(const std::string& filepath,
                              size_t& num_elements,
                              size_t& file_size,
                              std::string& data_type);

    // Helper: extract dataset name from filepath
    static std::string getDatasetName(const std::string& filepath);

    // Helper: check if file is binary format
    static bool isBinaryFormat(const std::string& filepath);
};

// Dataset metadata
struct SOSDDatasetInfo {
    std::string name;
    std::string filepath;
    size_t num_elements;
    size_t element_size;  // bytes
    std::string data_type;  // "uint32" or "uint64"
    bool is_binary;
};

#endif // SOSD_LOADER_H
