#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_txt> <output_bin>" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::ifstream in(input_file);
    if (!in.is_open()) {
        std::cerr << "Error opening input file: " << input_file << std::endl;
        return 1;
    }

    std::ofstream out(output_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error opening output file: " << output_file << std::endl;
        return 1;
    }

    uint32_t value;
    size_t count = 0;
    while (in >> value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(uint32_t));
        count++;
        if (count % 10000000 == 0) {
            std::cout << "Converted " << count / 1000000 << "M values..." << std::endl;
        }
    }

    in.close();
    out.close();

    std::cout << "Converted " << count << " values from " << input_file << " to " << output_file << std::endl;
    return 0;
}
