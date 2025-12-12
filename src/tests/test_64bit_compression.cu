/**
 * Test compilation of 64-bit GPU compression kernels
 */

#include <cstdint>
#include <iostream>

// Include all 64-bit headers
#include "third_party/VerticalGPU/Vertical/src/include/fls_gen/unpack/unpack_64.cuh"
#include "third_party/tile-gpu-compression/src/binpack_kernel_64.cuh"
#include "third_party/tile-gpu-compression/src/binpack_random_access_64.cuh"
#include "third_party/tile-gpu-compression/src/deltabinpack_kernel_64.cuh"
#include "third_party/tile-gpu-compression/src/deltabinpack_random_access_64.cuh"
#include "third_party/tile-gpu-compression/src/rlebinpack_kernel_64.cuh"

__global__ void test_Vertical_64() {
    __shared__ uint64_t input[1024];
    __shared__ uint64_t output[1024];

    // Test unpack function
    generated::unpack::cuda::normal_64::unpack(input, output, 16);
}

__global__ void test_binpack_64(
    uint64_t* compressed,
    int64_t* output,
    uint* block_starts)
{
    // Test binpack decode
    int64_t val = decodeElementAtIndex64(threadIdx.x, block_starts, compressed);
    output[threadIdx.x] = val;
}

int main() {
    std::cout << "64-bit GPU compression kernels compile test" << std::endl;
    std::cout << "All headers compiled successfully!" << std::endl;

    // Test can instantiate kernel launches
    test_Vertical_64<<<1, 32>>>();

    uint64_t* d_compressed;
    int64_t* d_output;
    uint* d_block_starts;

    cudaMalloc(&d_compressed, 1024 * sizeof(uint64_t));
    cudaMalloc(&d_output, 1024 * sizeof(int64_t));
    cudaMalloc(&d_block_starts, 128 * sizeof(uint));

    test_binpack_64<<<1, 128>>>(d_compressed, d_output, d_block_starts);

    cudaFree(d_compressed);
    cudaFree(d_output);
    cudaFree(d_block_starts);

    cudaDeviceSynchronize();

    std::cout << "Test completed!" << std::endl;
    return 0;
}
