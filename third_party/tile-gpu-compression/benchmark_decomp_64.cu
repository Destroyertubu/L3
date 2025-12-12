/**
 * Benchmark 64-bit decompression throughput for Tile-GPU-Compression
 * Tests FOR, DFOR, RFOR on pre-compressed 64-bit datasets
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
} while(0)

// Timing helper
class GpuTimer {
    cudaEvent_t start_, stop_;
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { cudaEventRecord(start_); }
    void stop() { cudaEventRecord(stop_); cudaEventSynchronize(stop_); }
    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// ============================================================================
// FOR 64-bit decode kernel
// ============================================================================
__global__ void decodeFOR64Kernel(
    const uint64_t* __restrict__ packed_data,
    int64_t* __restrict__ output,
    const uint32_t* __restrict__ block_starts,  // 32-bit offsets
    int64_t num_elements,
    int elements_per_block = 128)
{
    int block_idx = blockIdx.x;
    int64_t global_offset = (int64_t)block_idx * elements_per_block;
    if (global_offset >= num_elements) return;

    int tid = threadIdx.x;

    // Get this block's data offset
    uint32_t block_offset = block_starts[block_idx];
    const uint64_t* data_block = packed_data + block_offset;

    // Read reference (first 64-bit word)
    int64_t reference = reinterpret_cast<const int64_t*>(data_block)[0];

    // Read miniblock bitwidths (4 x 8-bit, stored after reference)
    // Layout: [64-bit ref][32-bit bitwidths padding to 64-bit]
    uint32_t miniblock_bitwidths = reinterpret_cast<const uint32_t*>(data_block + 1)[0];

    // Skip header to get to packed values
    const uint64_t* value_data = data_block + 2;  // Skip 16 bytes header

    // Decode elements - each thread handles one element in simple version
    int idx = tid;
    int items_this_block = min((int64_t)elements_per_block, num_elements - global_offset);

    while (idx < items_this_block) {
        // Determine which miniblock (0-3) and position within
        int miniblock_idx = idx / 32;
        int pos_in_miniblock = idx % 32;

        // Get bitwidth for this miniblock
        uint32_t bitwidth = (miniblock_bitwidths >> (miniblock_idx * 8)) & 0xFF;

        // Calculate bit offset for this miniblock's start
        uint32_t miniblock_bit_start = 0;
        for (int m = 0; m < miniblock_idx; m++) {
            uint32_t bw = (miniblock_bitwidths >> (m * 8)) & 0xFF;
            miniblock_bit_start += bw * 32;
        }

        // Bit position of this element
        uint32_t bit_pos = miniblock_bit_start + pos_in_miniblock * bitwidth;
        uint32_t word_idx = bit_pos / 64;
        uint32_t bit_offset = bit_pos % 64;

        uint64_t element;
        if (bitwidth == 0) {
            element = 0;
        } else if (bitwidth == 64) {
            element = value_data[word_idx];
        } else {
            uint64_t mask = (1ULL << bitwidth) - 1ULL;
            uint64_t lo = value_data[word_idx];

            if (bit_offset + bitwidth <= 64) {
                element = (lo >> bit_offset) & mask;
            } else {
                uint64_t hi = value_data[word_idx + 1];
                uint32_t bits_from_lo = 64 - bit_offset;
                element = (lo >> bit_offset) | ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
            }
        }

        output[global_offset + idx] = reference + static_cast<int64_t>(element);
        idx += blockDim.x;
    }
}

// ============================================================================
// DFOR 64-bit decode kernel (Delta + FOR)
// ============================================================================
__global__ void decodeDFOR64Kernel(
    const uint64_t* __restrict__ packed_data,
    int64_t* __restrict__ output,
    const uint32_t* __restrict__ block_starts,
    int64_t num_elements,
    int elements_per_block = 128)
{
    int block_idx = blockIdx.x;
    int64_t global_offset = (int64_t)block_idx * elements_per_block;
    if (global_offset >= num_elements) return;

    int tid = threadIdx.x;

    __shared__ int64_t shared_vals[128];

    uint32_t block_offset = block_starts[block_idx];
    const uint64_t* data_block = packed_data + block_offset;

    // For DFOR: reference is the first actual value, deltas are stored
    int64_t reference = reinterpret_cast<const int64_t*>(data_block)[0];
    uint32_t miniblock_bitwidths = reinterpret_cast<const uint32_t*>(data_block + 1)[0];
    const uint64_t* value_data = data_block + 2;

    int items_this_block = min((int64_t)elements_per_block, num_elements - global_offset);

    // First, decode the deltas (same as FOR)
    int idx = tid;
    while (idx < items_this_block) {
        int miniblock_idx = idx / 32;
        int pos_in_miniblock = idx % 32;
        uint32_t bitwidth = (miniblock_bitwidths >> (miniblock_idx * 8)) & 0xFF;

        uint32_t miniblock_bit_start = 0;
        for (int m = 0; m < miniblock_idx; m++) {
            uint32_t bw = (miniblock_bitwidths >> (m * 8)) & 0xFF;
            miniblock_bit_start += bw * 32;
        }

        uint32_t bit_pos = miniblock_bit_start + pos_in_miniblock * bitwidth;
        uint32_t word_idx = bit_pos / 64;
        uint32_t bit_offset = bit_pos % 64;

        int64_t delta;
        if (bitwidth == 0) {
            delta = 0;
        } else if (bitwidth == 64) {
            delta = value_data[word_idx];
        } else {
            uint64_t mask = (1ULL << bitwidth) - 1ULL;
            uint64_t lo = value_data[word_idx];

            if (bit_offset + bitwidth <= 64) {
                delta = (lo >> bit_offset) & mask;
            } else {
                uint64_t hi = value_data[word_idx + 1];
                uint32_t bits_from_lo = 64 - bit_offset;
                delta = (lo >> bit_offset) | ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
            }
        }

        shared_vals[idx] = delta;
        idx += blockDim.x;
    }
    __syncthreads();

    // Prefix sum to convert deltas to values
    // Simple sequential prefix sum (could be optimized with parallel scan)
    if (tid == 0) {
        int64_t running_sum = reference;
        for (int i = 0; i < items_this_block; i++) {
            running_sum += shared_vals[i];
            output[global_offset + i] = running_sum;
        }
    }
}

// ============================================================================
// RFOR 64-bit decode kernel (RLE + FOR) - simplified
// ============================================================================
__global__ void decodeRFOR64Kernel(
    const uint64_t* __restrict__ val_data,
    const uint64_t* __restrict__ rl_data,
    int64_t* __restrict__ output,
    const uint32_t* __restrict__ val_starts,
    const uint32_t* __restrict__ rl_starts,
    int64_t num_elements,
    int elements_per_block = 128)
{
    // RFOR is more complex - simplified implementation
    // For now, just output zeros to measure kernel launch overhead
    int block_idx = blockIdx.x;
    int64_t global_offset = (int64_t)block_idx * elements_per_block;
    if (global_offset >= num_elements) return;

    int tid = threadIdx.x;
    int items_this_block = min((int64_t)elements_per_block, num_elements - global_offset);

    // Placeholder - real RFOR decoding would expand run-lengths
    for (int idx = tid; idx < items_this_block; idx += blockDim.x) {
        output[global_offset + idx] = 0;
    }
}

// ============================================================================
// Test Functions
// ============================================================================

struct CompressedData {
    vector<uint64_t> data;
    vector<uint32_t> offsets;
    int64_t num_elements;
    size_t compressed_size;
};

CompressedData loadCompressed(const string& data_file, const string& offset_file) {
    CompressedData result;

    // Load data file
    ifstream df(data_file, ios::binary | ios::ate);
    if (!df) {
        cerr << "Failed to open: " << data_file << endl;
        return result;
    }
    size_t data_size = df.tellg();
    df.seekg(0);
    result.data.resize(data_size / sizeof(uint64_t));
    df.read(reinterpret_cast<char*>(result.data.data()), data_size);
    result.compressed_size = data_size;

    // Load offset file
    ifstream of(offset_file, ios::binary | ios::ate);
    if (!of) {
        cerr << "Failed to open: " << offset_file << endl;
        return result;
    }
    size_t offset_size = of.tellg();
    of.seekg(0);
    result.offsets.resize(offset_size / sizeof(uint32_t));
    of.read(reinterpret_cast<char*>(result.offsets.data()), offset_size);

    // num_elements = (num_blocks - 1) * 128
    result.num_elements = ((int64_t)result.offsets.size() - 1) * 128;

    return result;
}

double benchmarkFOR64(const string& prefix, int64_t expected_elements, int num_trials = 10) {
    string data_file = prefix + "_FOR_compressed_64.bin";
    string offset_file = prefix + "_FOR_compressed_64.binoff";

    CompressedData comp = loadCompressed(data_file, offset_file);
    if (comp.data.empty()) return -1;

    // Allocate GPU memory
    uint64_t* d_data;
    uint32_t* d_offsets;
    int64_t* d_output;

    CUDA_CHECK(cudaMalloc(&d_data, comp.data.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_offsets, comp.offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output, comp.num_elements * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_data, comp.data.data(), comp.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, comp.offsets.data(), comp.offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int num_blocks = (comp.num_elements + 127) / 128;

    // Warmup
    decodeFOR64Kernel<<<num_blocks, 128>>>(d_data, d_output, d_offsets, comp.num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    timer.start();
    for (int t = 0; t < num_trials; t++) {
        decodeFOR64Kernel<<<num_blocks, 128>>>(d_data, d_output, d_offsets, comp.num_elements);
    }
    timer.stop();

    double avg_ms = timer.elapsed_ms() / num_trials;
    double throughput_gbps = (comp.num_elements * sizeof(int64_t) / 1e9) / (avg_ms / 1000.0);

    cout << "  FOR-64: " << avg_ms << " ms, " << throughput_gbps << " GB/s" << endl;
    cout << "    Compressed: " << comp.compressed_size / 1e9 << " GB, Elements: " << comp.num_elements << endl;
    cout << "    Ratio: " << (comp.num_elements * 8.0) / comp.compressed_size << "x" << endl;

    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_output);

    return throughput_gbps;
}

double benchmarkDFOR64(const string& prefix, int64_t expected_elements, int num_trials = 10) {
    string data_file = prefix + "_DFOR_compressed_dfor_64.bin";
    string offset_file = prefix + "_DFOR_compressed_dfor_64.binoff";

    CompressedData comp = loadCompressed(data_file, offset_file);
    if (comp.data.empty()) return -1;

    uint64_t* d_data;
    uint32_t* d_offsets;
    int64_t* d_output;

    CUDA_CHECK(cudaMalloc(&d_data, comp.data.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_offsets, comp.offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output, comp.num_elements * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_data, comp.data.data(), comp.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, comp.offsets.data(), comp.offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int num_blocks = (comp.num_elements + 127) / 128;

    // Warmup
    decodeDFOR64Kernel<<<num_blocks, 128>>>(d_data, d_output, d_offsets, comp.num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    timer.start();
    for (int t = 0; t < num_trials; t++) {
        decodeDFOR64Kernel<<<num_blocks, 128>>>(d_data, d_output, d_offsets, comp.num_elements);
    }
    timer.stop();

    double avg_ms = timer.elapsed_ms() / num_trials;
    double throughput_gbps = (comp.num_elements * sizeof(int64_t) / 1e9) / (avg_ms / 1000.0);

    cout << "  DFOR-64: " << avg_ms << " ms, " << throughput_gbps << " GB/s" << endl;
    cout << "    Compressed: " << comp.compressed_size / 1e9 << " GB, Elements: " << comp.num_elements << endl;
    cout << "    Ratio: " << (comp.num_elements * 8.0) / comp.compressed_size << "x" << endl;

    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_output);

    return throughput_gbps;
}

int main(int argc, char** argv) {
    cout << "=== Tile-GPU-Compression 64-bit Decompression Benchmark ===" << endl;
    cout << "Test Platform: NVIDIA H20" << endl << endl;

    string tile_dir = "/root/autodl-tmp/test/data/sosd/tile_compressed/";

    // Dataset 8: OSM
    cout << "Dataset 8: OSM CellIDs (800M uint64)" << endl;
    string osm_prefix = tile_dir + "8-osm_cellids_800M_uint64";
    benchmarkFOR64(osm_prefix, 800000000);
    benchmarkDFOR64(osm_prefix, 800000000);
    cout << endl;

    // Dataset 13: Medicare
    cout << "Dataset 13: Medicare (800M uint64 sample)" << endl;
    string med_prefix = tile_dir + "13-medicare_800M";
    benchmarkFOR64(med_prefix, 800000000);
    benchmarkDFOR64(med_prefix, 800000000);

    cout << endl << "Benchmark complete." << endl;

    return 0;
}
