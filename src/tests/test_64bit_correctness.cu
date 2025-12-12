/**
 * Correctness test for 64-bit GPU compression kernels
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>

#include "third_party/VerticalGPU/Vertical/src/include/fls_gen/unpack/unpack_64.cuh"
#include "third_party/tile-gpu-compression/src/binpack_kernel_64.cuh"
#include "third_party/tile-gpu-compression/src/binpack_random_access_64.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Test Vertical 64-bit unpacking
__global__ void test_unpack_kernel(
    const uint64_t* __restrict__ packed,
    uint64_t* __restrict__ unpacked,
    uint8_t bitwidth)
{
    generated::unpack::cuda::normal_64::unpack(packed, unpacked, bitwidth);
}

// Host-side packing function for reference
void pack_values_host(const uint64_t* values, uint64_t* packed, int count, int bitwidth) {
    if (bitwidth == 0) return;
    if (bitwidth == 64) {
        memcpy(packed, values, count * sizeof(uint64_t));
        return;
    }

    memset(packed, 0, ((count * bitwidth + 63) / 64) * sizeof(uint64_t));
    uint64_t mask = (1ULL << bitwidth) - 1;

    for (int i = 0; i < count; i++) {
        uint64_t val = values[i] & mask;
        int bit_pos = i * bitwidth;
        int word_idx = bit_pos / 64;
        int bit_offset = bit_pos % 64;

        packed[word_idx] |= (val << bit_offset);
        if (bit_offset + bitwidth > 64) {
            packed[word_idx + 1] |= (val >> (64 - bit_offset));
        }
    }
}

bool test_Vertical_bitwidth(int bitwidth) {
    const int NUM_VALUES = 1024;  // Vertical processes 1024 values per block
    const int PACKED_WORDS = (NUM_VALUES * bitwidth + 63) / 64;

    std::vector<uint64_t> h_values(NUM_VALUES);
    std::vector<uint64_t> h_packed(PACKED_WORDS + 1);  // +1 for safety
    std::vector<uint64_t> h_unpacked(NUM_VALUES);

    // Generate test values
    uint64_t max_val = (bitwidth == 64) ? UINT64_MAX : ((1ULL << bitwidth) - 1);
    for (int i = 0; i < NUM_VALUES; i++) {
        h_values[i] = (uint64_t)(i * 7 + 13) % (max_val + 1);
    }

    // Pack on host
    pack_values_host(h_values.data(), h_packed.data(), NUM_VALUES, bitwidth);

    // Allocate device memory
    uint64_t *d_packed, *d_unpacked;
    CHECK_CUDA(cudaMalloc(&d_packed, (PACKED_WORDS + 1) * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_unpacked, NUM_VALUES * sizeof(uint64_t)));

    CHECK_CUDA(cudaMemcpy(d_packed, h_packed.data(), PACKED_WORDS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_unpacked, 0, NUM_VALUES * sizeof(uint64_t)));

    // Run unpack kernel
    test_unpack_kernel<<<1, 32>>>(d_packed, d_unpacked, bitwidth);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and verify
    CHECK_CUDA(cudaMemcpy(h_unpacked.data(), d_unpacked, NUM_VALUES * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < NUM_VALUES && errors < 10; i++) {
        if (h_unpacked[i] != h_values[i]) {
            std::cerr << "  Mismatch at index " << i << ": expected " << h_values[i]
                      << ", got " << h_unpacked[i] << std::endl;
            errors++;
        }
    }

    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_unpacked));

    return errors == 0;
}

// Test binpack random access
__global__ void test_random_access_kernel(
    uint* block_start,
    uint64_t* compressed,
    uint* indices,
    int64_t* output,
    int num_queries)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_queries) {
        output[tid] = decodeElementAtIndex64(indices[tid], block_start, compressed);
    }
}

bool test_random_access() {
    // Create a simple compressed block with known values
    // Block format: [reference (8 bytes)][bitwidths (4 bytes, padded to 8)][packed data]
    const int NUM_ELEMENTS = 128;
    const int BITWIDTH = 16;
    const int64_t REFERENCE = 1000;

    std::vector<uint64_t> h_compressed(32);  // Enough for header + data
    std::vector<uint> h_block_start = {0};

    // Set reference
    reinterpret_cast<int64_t*>(h_compressed.data())[0] = REFERENCE;

    // Set bitwidths (4 miniblocks, each with BITWIDTH bits)
    uint bitwidths = BITWIDTH | (BITWIDTH << 8) | (BITWIDTH << 16) | (BITWIDTH << 24);
    reinterpret_cast<uint*>(h_compressed.data())[2] = bitwidths;

    // Pack values (simple: each element is its index)
    // Data starts at offset 2 (after header)
    uint64_t* data_ptr = h_compressed.data() + 2;
    for (int miniblock = 0; miniblock < 4; miniblock++) {
        int miniblock_offset = 0;
        for (int mb = 0; mb < miniblock; mb++) {
            miniblock_offset += (32 * BITWIDTH + 63) / 64;
        }

        for (int i = 0; i < 32; i++) {
            int global_idx = miniblock * 32 + i;
            uint64_t val = global_idx;  // Simple test value

            int bit_pos = i * BITWIDTH;
            int word_idx = bit_pos / 64;
            int bit_offset = bit_pos % 64;

            data_ptr[miniblock_offset + word_idx] |= (val << bit_offset);
            if (bit_offset + BITWIDTH > 64) {
                data_ptr[miniblock_offset + word_idx + 1] |= (val >> (64 - bit_offset));
            }
        }
    }

    // Test random access
    std::vector<uint> h_indices = {0, 31, 32, 63, 64, 95, 96, 127};
    std::vector<int64_t> h_output(h_indices.size());

    uint64_t *d_compressed;
    uint *d_block_start, *d_indices;
    int64_t *d_output;

    CHECK_CUDA(cudaMalloc(&d_compressed, h_compressed.size() * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_block_start, h_block_start.size() * sizeof(uint)));
    CHECK_CUDA(cudaMalloc(&d_indices, h_indices.size() * sizeof(uint)));
    CHECK_CUDA(cudaMalloc(&d_output, h_indices.size() * sizeof(int64_t)));

    CHECK_CUDA(cudaMemcpy(d_compressed, h_compressed.data(), h_compressed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_block_start, h_block_start.data(), h_block_start.size() * sizeof(uint), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(uint), cudaMemcpyHostToDevice));

    test_random_access_kernel<<<1, 128>>>(d_block_start, d_compressed, d_indices, d_output, h_indices.size());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, h_indices.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < h_indices.size(); i++) {
        int64_t expected = REFERENCE + h_indices[i];
        if (h_output[i] != expected) {
            std::cerr << "  Random access mismatch at query " << i << " (index " << h_indices[i]
                      << "): expected " << expected << ", got " << h_output[i] << std::endl;
            errors++;
        }
    }

    CHECK_CUDA(cudaFree(d_compressed));
    CHECK_CUDA(cudaFree(d_block_start));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_output));

    return errors == 0;
}

int main() {
    std::cout << "=== 64-bit GPU Compression Correctness Tests ===" << std::endl;

    // Test Vertical unpacking for various bit widths
    std::cout << "\n1. Vertical 64-bit Unpack Tests:" << std::endl;
    int bitwidths_to_test[] = {0, 1, 2, 4, 8, 16, 32, 64};
    int pass_count = 0;
    int total_count = sizeof(bitwidths_to_test) / sizeof(bitwidths_to_test[0]);

    for (int bw : bitwidths_to_test) {
        bool passed = test_Vertical_bitwidth(bw);
        std::cout << "   Bitwidth " << bw << ": " << (passed ? "PASS" : "FAIL") << std::endl;
        if (passed) pass_count++;
    }
    std::cout << "   Vertical: " << pass_count << "/" << total_count << " tests passed" << std::endl;

    // Test random access
    std::cout << "\n2. Binpack Random Access Test:" << std::endl;
    bool ra_passed = test_random_access();
    std::cout << "   Random Access: " << (ra_passed ? "PASS" : "FAIL") << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    bool all_passed = (pass_count == total_count) && ra_passed;
    std::cout << "Overall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_passed ? 0 : 1;
}
