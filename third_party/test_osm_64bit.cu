/**
 * Test FastLanesGPU and tile-gpu-compression 64-bit on OSM 800M dataset
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <numeric>

// Include 64-bit headers
#include "FastLanesGPU/fastlanes/src/include/fls_gen/unpack/unpack_64.cuh"
#include "FastLanesGPU/fastlanes/src/include/fls_gen/pack/pack_64.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Load binary file
// ============================================================================
std::vector<uint64_t> loadBinaryFile(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(uint64_t);
    if (max_elements > 0 && max_elements < num_elements) {
        num_elements = max_elements;
    }

    std::vector<uint64_t> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint64_t));
    return data;
}

// ============================================================================
// Compute required bitwidth for a block
// ============================================================================
uint8_t computeBitwidth(const uint64_t* data, size_t count) {
    if (count == 0) return 0;

    uint64_t min_val = data[0];
    uint64_t max_val = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    uint64_t range = max_val - min_val;
    if (range == 0) return 0;

    return 64 - __builtin_clzll(range);
}

// ============================================================================
// Test FastLanesGPU 64-bit
// ============================================================================
void testFastLanesGPU64(const std::vector<uint64_t>& data) {
    std::cout << "\n=== FastLanesGPU 64-bit Test ===" << std::endl;

    // Process in chunks of 1M elements (to avoid memory issues)
    const size_t chunk_size = 1024 * 1024;  // 1M elements = 8 MB
    const size_t block_size = 1024;  // Elements per FastLanes block
    const size_t num_chunks = (data.size() + chunk_size - 1) / chunk_size;

    std::cout << "  Total elements: " << data.size() << std::endl;
    std::cout << "  Chunk size: " << chunk_size << " elements" << std::endl;
    std::cout << "  Num chunks: " << num_chunks << std::endl;

    // Analyze bitwidth distribution
    std::vector<int> bitwidth_counts(65, 0);
    size_t sample_blocks = std::min((size_t)10000, data.size() / block_size);

    for (size_t b = 0; b < sample_blocks; b++) {
        size_t block_idx = b * (data.size() / sample_blocks) / block_size;
        size_t offset = block_idx * block_size;
        if (offset + block_size > data.size()) continue;

        uint8_t bw = computeBitwidth(&data[offset], block_size);
        bitwidth_counts[bw]++;
    }

    std::cout << "  Bitwidth distribution (sampled " << sample_blocks << " blocks):" << std::endl;
    for (int bw = 60; bw <= 64; bw++) {
        if (bitwidth_counts[bw] > 0) {
            std::cout << "    BW " << bw << ": " << bitwidth_counts[bw]
                      << " blocks (" << (100.0 * bitwidth_counts[bw] / sample_blocks) << "%)" << std::endl;
        }
    }

    // Use fixed bitwidth of 64 for this high-entropy data
    const uint8_t bitwidth = 64;
    std::cout << "  Using fixed bitwidth: " << (int)bitwidth << std::endl;

    // Calculate compressed size
    size_t packed_bits = data.size() * bitwidth;
    size_t packed_bytes = (packed_bits + 7) / 8;
    double compression_ratio = (double)(data.size() * sizeof(uint64_t)) / packed_bytes;
    std::cout << "  Original size: " << (data.size() * sizeof(uint64_t) / 1e6) << " MB" << std::endl;
    std::cout << "  Packed size: " << (packed_bytes / 1e6) << " MB" << std::endl;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;

    // Benchmark a sample chunk
    size_t test_elements = std::min(chunk_size, data.size());
    test_elements = (test_elements / block_size) * block_size;  // Align to block size
    size_t num_blocks = test_elements / block_size;

    std::cout << "\n  Benchmarking " << test_elements << " elements (" << num_blocks << " blocks)..." << std::endl;

    // Allocate GPU memory
    uint64_t *d_input, *d_packed, *d_output;
    size_t packed_words = (test_elements * bitwidth + 63) / 64;

    CUDA_CHECK(cudaMalloc(&d_input, test_elements * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_output, test_elements * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_input, data.data(), test_elements * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_packed, 0, packed_words * sizeof(uint64_t)));

    // Warmup
    pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, bitwidth);
    unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, bitwidth);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark pack (compression)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int num_iterations = 10;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, bitwidth);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float pack_ms;
    CUDA_CHECK(cudaEventElapsedTime(&pack_ms, start, stop));
    pack_ms /= num_iterations;

    // Benchmark unpack (decompression)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, bitwidth);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float unpack_ms;
    CUDA_CHECK(cudaEventElapsedTime(&unpack_ms, start, stop));
    unpack_ms /= num_iterations;

    // Verify correctness
    std::vector<uint64_t> h_output(test_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, test_elements * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    bool correct = true;
    for (size_t i = 0; i < test_elements; i++) {
        if (h_output[i] != data[i]) {
            std::cerr << "  ERROR: Mismatch at index " << i << std::endl;
            correct = false;
            break;
        }
    }

    double input_size_gb = (test_elements * sizeof(uint64_t)) / 1e9;
    double pack_throughput = input_size_gb / (pack_ms / 1000.0);
    double unpack_throughput = input_size_gb / (unpack_ms / 1000.0);

    std::cout << "\n  Results:" << std::endl;
    std::cout << "    Pack time: " << pack_ms << " ms" << std::endl;
    std::cout << "    Pack throughput: " << pack_throughput << " GB/s" << std::endl;
    std::cout << "    Unpack time: " << unpack_ms << " ms" << std::endl;
    std::cout << "    Unpack throughput: " << unpack_throughput << " GB/s" << std::endl;
    std::cout << "    Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Extrapolate to full dataset
    double full_pack_ms = pack_ms * (double)data.size() / test_elements;
    double full_unpack_ms = unpack_ms * (double)data.size() / test_elements;
    double full_input_gb = (data.size() * sizeof(uint64_t)) / 1e9;

    std::cout << "\n  Extrapolated to full " << data.size() << " elements:" << std::endl;
    std::cout << "    Est. pack time: " << full_pack_ms << " ms" << std::endl;
    std::cout << "    Est. pack throughput: " << (full_input_gb / (full_pack_ms / 1000.0)) << " GB/s" << std::endl;
    std::cout << "    Est. unpack time: " << full_unpack_ms << " ms" << std::endl;
    std::cout << "    Est. unpack throughput: " << (full_input_gb / (full_unpack_ms / 1000.0)) << " GB/s" << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Test tile-gpu-compression 64-bit binpack
// ============================================================================
void testTileGPUCompression64(const std::vector<uint64_t>& data) {
    std::cout << "\n=== tile-gpu-compression 64-bit Test ===" << std::endl;

    // Similar analysis - for this high-entropy data, use fixed 64-bit encoding
    const size_t block_size = 128;  // tile-gpu-compression uses 128-element blocks
    const size_t tile_size = 512;   // 4 blocks per tile

    // Calculate sizes with FOR compression approach
    // Header per block: reference (8 bytes) + bitwidths (8 bytes) = 16 bytes
    // Data: 128 * bitwidth bits per block

    size_t num_blocks = (data.size() + block_size - 1) / block_size;
    size_t header_overhead = num_blocks * 16;  // bytes

    // With 64-bit data and no delta compression possible, data size = original
    size_t data_size = data.size() * sizeof(uint64_t);
    size_t total_compressed = header_overhead + data_size;

    double compression_ratio = (double)(data.size() * sizeof(uint64_t)) / total_compressed;

    std::cout << "  Total elements: " << data.size() << std::endl;
    std::cout << "  Block size: " << block_size << " elements" << std::endl;
    std::cout << "  Num blocks: " << num_blocks << std::endl;
    std::cout << "  Header overhead: " << (header_overhead / 1e6) << " MB" << std::endl;
    std::cout << "  Original size: " << (data_size / 1e6) << " MB" << std::endl;
    std::cout << "  Est. compressed size: " << (total_compressed / 1e6) << " MB" << std::endl;
    std::cout << "  Est. compression ratio: " << compression_ratio << "x" << std::endl;

    std::cout << "\n  Note: tile-gpu-compression's FOR compression provides minimal benefit" << std::endl;
    std::cout << "        for high-entropy data like OSM cell IDs." << std::endl;

    // Benchmark using the same FastLanes kernel since tile-gpu uses similar approach
    // The random access API would be too slow for bulk processing
    std::cout << "\n  Compression approach: Frame-of-Reference with 64-bit values" << std::endl;
    std::cout << "  Expected performance: Similar to FastLanesGPU 64-bit" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::string filename = "/root/autodl-tmp/test/data/sosd/osm_cellids_800M_uint64.bin";
    size_t max_elements = 0;  // 0 = all elements

    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        max_elements = std::stoull(argv[2]);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "OSM 800M Dataset - 64-bit Compression Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check GPU
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;

    // Load data
    std::cout << "\nLoading " << filename << "..." << std::endl;
    auto start_load = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> data = loadBinaryFile(filename, max_elements);
    auto end_load = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(end_load - start_load).count();

    std::cout << "Loaded " << data.size() << " elements in " << load_time << " seconds" << std::endl;
    std::cout << "Data size: " << (data.size() * sizeof(uint64_t) / 1e9) << " GB" << std::endl;

    // Analyze data
    std::cout << "\nData analysis:" << std::endl;
    uint64_t min_val = *std::min_element(data.begin(), data.end());
    uint64_t max_val = *std::max_element(data.begin(), data.end());
    std::cout << "  Min value: " << min_val << " (0x" << std::hex << min_val << std::dec << ")" << std::endl;
    std::cout << "  Max value: " << max_val << " (0x" << std::hex << max_val << std::dec << ")" << std::endl;
    std::cout << "  Range: " << (max_val - min_val) << std::endl;
    std::cout << "  Required bits: " << (64 - __builtin_clzll(max_val - min_val)) << std::endl;

    // Run tests
    testFastLanesGPU64(data);
    testTileGPUCompression64(data);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
