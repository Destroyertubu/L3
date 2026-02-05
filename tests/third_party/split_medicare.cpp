#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <cstring>

int main() {
    const char* input_file = "/root/autodl-tmp/code/L3/data/sosd/13-medicare.bin";
    const char* output_dir = "/root/autodl-tmp/code/L3/data/sosd/";

    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open input file: " << input_file << std::endl;
        return 1;
    }

    // Get file size
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    // Read header (8 bytes = count)
    uint64_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Element count: " << count << std::endl;

    // Split into 4 parts (each ~3GB)
    const int num_parts = 4;
    size_t elements_per_part = count / num_parts;

    std::vector<uint64_t> buffer(elements_per_part + 1);

    for (int part = 0; part < num_parts; part++) {
        size_t start = part * elements_per_part;
        size_t end = (part == num_parts - 1) ? count : (part + 1) * elements_per_part;
        size_t part_count = end - start;

        // Read elements
        in.seekg(8 + start * sizeof(uint64_t), std::ios::beg);
        in.read(reinterpret_cast<char*>(buffer.data()), part_count * sizeof(uint64_t));

        // Write output file
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "%s13-medicare_part%d.bin", output_dir, part + 1);

        std::ofstream out(output_file, std::ios::binary);
        if (!out) {
            std::cerr << "Cannot create output file: " << output_file << std::endl;
            return 1;
        }

        // Write header (count for this part)
        out.write(reinterpret_cast<const char*>(&part_count), sizeof(part_count));
        // Write data
        out.write(reinterpret_cast<const char*>(buffer.data()), part_count * sizeof(uint64_t));
        out.close();

        std::cout << "Created: " << output_file << " (" << part_count << " elements, "
                  << (part_count * 8 + 8) / (1024*1024) << " MB)" << std::endl;
    }

    in.close();
    std::cout << "Done!" << std::endl;
    return 0;
}
