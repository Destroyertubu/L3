/**
 * Basic GLECO Compression Example
 *
 * Demonstrates:
 * - Creating sample data
 * - Compressing with GLECO
 * - Decompressing and verifying
 * - Printing statistics
 */

#include "L3_codec.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== GLECO Basic Compression Example ===" << std::endl;

    // 1. Create sample data (linear pattern for good compression)
    const int N = 1000000;
    std::vector<uint32_t> data(N);

    std::cout << "\nGenerating " << N << " elements..." << std::endl;
    for (int i = 0; i < N; i++) {
        data[i] = 1000000 + i * 10;  // Linear pattern: good for learned compression
    }

    // 2. Compress the data
    std::cout << "Compressing data..." << std::endl;

    CompressionStats comp_stats;
    auto compressed = compressData(data, 1024, &comp_stats);

    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        return 1;
    }

    // 3. Print compression statistics
    std::cout << "\n=== Compression Statistics ===" << std::endl;
    std::cout << "Original size:    " << std::setw(10) << comp_stats.original_size_bytes
              << " bytes (" << (comp_stats.original_size_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Compressed size:  " << std::setw(10) << comp_stats.compressed_size_bytes
              << " bytes (" << (comp_stats.compressed_size_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
              << comp_stats.compression_ratio << "x" << std::endl;
    std::cout << "Compression time:  " << std::fixed << std::setprecision(3)
              << comp_stats.compression_time_ms << " ms" << std::endl;
    std::cout << "Num partitions:    " << comp_stats.num_partitions << std::endl;
    std::cout << "Avg delta bits:    " << std::fixed << std::setprecision(2)
              << comp_stats.avg_delta_bits << std::endl;

    // 4. Decompress the data
    std::cout << "\nDecompressing data..." << std::endl;

    std::vector<uint32_t> decompressed;
    DecompressionStats decomp_stats;

    int ret = decompressData(compressed, decompressed, &decomp_stats);
    if (ret != 0) {
        std::cerr << "Decompression failed!" << std::endl;
        freeCompressedData(compressed);
        return 1;
    }

    std::cout << "Decompression time: " << std::fixed << std::setprecision(3)
              << decomp_stats.decompression_time_ms << " ms" << std::endl;
    std::cout << "Throughput:         " << std::fixed << std::setprecision(2)
              << (comp_stats.original_size_bytes / 1024.0 / 1024.0 / (decomp_stats.decompression_time_ms / 1000.0))
              << " MB/s" << std::endl;

    // 5. Verify correctness
    std::cout << "\nVerifying correctness..." << std::endl;

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (data[i] != decompressed[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                      << data[i] << " != " << decompressed[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "✓ All " << N << " elements match!" << std::endl;
    } else {
        std::cout << "✗ Verification failed!" << std::endl;
    }

    // 6. Cleanup
    freeCompressedData(compressed);

    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}
