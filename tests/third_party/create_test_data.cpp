#include <fstream>
#include <cstdint>
#include <cstdlib>

int main() {
    const size_t n = 10240;  // 10 vectors of 1024 values
    uint32_t* data = new uint32_t[n];
    
    // Generate sorted data (good for delta encoding)
    for (size_t i = 0; i < n; i++) {
        data[i] = (uint32_t)(i * 3 + (rand() % 10));
    }
    
    std::ofstream file("test_data_uint32.bin", std::ios::binary);
    uint64_t count = n;
    file.write(reinterpret_cast<char*>(&count), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(data), n * sizeof(uint32_t));
    file.close();
    
    delete[] data;
    return 0;
}
