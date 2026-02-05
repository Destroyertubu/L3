#include <fstream>
#include <cstdint>
#include <cstdlib>

int main() {
    const size_t n = 10240;  // 10 vectors of 1024 values
    uint64_t* data = new uint64_t[n];

    // Generate sorted data (good for delta encoding)
    // Use larger values to test 64-bit handling
    for (size_t i = 0; i < n; i++) {
        data[i] = (uint64_t)(i * 3 + (rand() % 10)) + (1ULL << 40);
    }

    std::ofstream file("test_data_uint64.bin", std::ios::binary);
    uint64_t count = n;
    file.write(reinterpret_cast<char*>(&count), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(data), n * sizeof(uint64_t));
    file.close();

    delete[] data;
    return 0;
}
