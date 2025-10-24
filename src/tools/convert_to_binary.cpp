/**
 * Convert text files to binary format for fair benchmarking
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void convertTextToBinary(const std::string& input_file, const std::string& output_file, bool is_uint64 = false) {
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open input file: " + input_file);
    }

    if (is_uint64) {
        std::vector<uint64_t> data;
        uint64_t value;
        while (infile >> value) {
            data.push_back(value);
        }
        infile.close();

        std::ofstream outfile(output_file, std::ios::binary);
        if (!outfile.is_open()) {
            throw std::runtime_error("Cannot open output file: " + output_file);
        }
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint64_t));
        outfile.close();

        std::cout << "Converted " << data.size() << " uint64 elements to " << output_file << std::endl;
    } else {
        std::vector<uint32_t> data;
        uint32_t value;
        while (infile >> value) {
            data.push_back(value);
        }
        infile.close();

        std::ofstream outfile(output_file, std::ios::binary);
        if (!outfile.is_open()) {
            throw std::runtime_error("Cannot open output file: " + output_file);
        }
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
        outfile.close();

        std::cout << "Converted " << data.size() << " uint32 elements to " << output_file << std::endl;
    }
}

int main() {
    try {
        // Convert text files to binary
        std::cout << "Converting text files to binary format..." << std::endl;

        // movieid.txt -> movieid_uint32.bin
        convertTextToBinary("/root/autodl-tmp/test/data/movieid.txt",
                          "/root/autodl-tmp/test/data/movieid_uint32.bin", false);

        // linear_200M_uint32.txt -> linear_200M_uint32_binary.bin
        convertTextToBinary("/root/autodl-tmp/test/data/linear_200M_uint32.txt",
                          "/root/autodl-tmp/test/data/linear_200M_uint32_binary.bin", false);

        // normal_200M_uint32.txt -> normal_200M_uint32_binary.bin
        convertTextToBinary("/root/autodl-tmp/test/data/normal_200M_uint32.txt",
                          "/root/autodl-tmp/test/data/normal_200M_uint32_binary.bin", false);

        std::cout << "All conversions completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}