/**
 * FastLanesGPU 64-bit Delta + BitPack Benchmark
 *
 * Benchmarks 64-bit delta encoding with bit-packing compression.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>

// Block configuration for 64-bit FastLanes
constexpr int BLOCK_SIZE_64 = 1024;  // 1024 values per block
constexpr int THREADS_PER_BLOCK = 32;
constexpr int VALUES_PER_THREAD = 32;

// CPU-side 64-bit delta + bit-packing encoder
uint64_t delta_bin_pack_64(const uint64_t* in, uint64_t* out, uint32_t* block_offsets, size_t num_entries) {
    uint64_t out_ofs = 0;

    constexpr int block_size = BLOCK_SIZE_64;

    // Header
    out[0] = block_size;
    out[1] = num_entries;
    out_ofs = 2;

    size_t num_blocks = (num_entries + block_size - 1) / block_size;

    for (size_t blk = 0; blk < num_blocks; blk++) {
        block_offsets[blk] = out_ofs;

        size_t blk_start = blk * block_size;
        size_t blk_end = std::min(blk_start + block_size, num_entries);
        size_t actual_size = blk_end - blk_start;

        // Store first value of block
        uint64_t first_val = in[blk_start];
        out[out_ofs++] = first_val;

        // Compute deltas
        int64_t* deltas = new int64_t[actual_size];
        deltas[0] = 0;
        for (size_t i = 1; i < actual_size; i++) {
            deltas[i] = (int64_t)in[blk_start + i] - (int64_t)in[blk_start + i - 1];
        }

        // Find min delta (to make all values positive)
        int64_t min_delta = 0;
        for (size_t i = 0; i < actual_size; i++) {
            if (deltas[i] < min_delta) min_delta = deltas[i];
        }

        // Store min_delta (as int64_t)
        out[out_ofs++] = (uint64_t)min_delta;

        // Calculate max bitwidth needed for adjusted deltas
        uint32_t max_bitwidth = 0;
        for (size_t i = 0; i < actual_size; i++) {
            uint64_t adjusted = (uint64_t)(deltas[i] - min_delta);
            if (adjusted > 0) {
                uint32_t bw = 64 - __builtin_clzll(adjusted);
                if (bw > max_bitwidth) max_bitwidth = bw;
            }
        }
        if (max_bitwidth == 0) max_bitwidth = 1;

        // Store bitwidth
        out[out_ofs++] = max_bitwidth;

        // Pack adjusted deltas
        uint64_t current_word = 0;
        int bit_pos = 0;

        for (size_t i = 0; i < actual_size; i++) {
            uint64_t adjusted = (uint64_t)(deltas[i] - min_delta);

            if (bit_pos + max_bitwidth <= 64) {
                current_word |= (adjusted << bit_pos);
                bit_pos += max_bitwidth;

                if (bit_pos == 64) {
                    out[out_ofs++] = current_word;
                    current_word = 0;
                    bit_pos = 0;
                }
            } else {
                // Cross word boundary
                current_word |= (adjusted << bit_pos);
                out[out_ofs++] = current_word;

                int bits_written = 64 - bit_pos;
                current_word = adjusted >> bits_written;
                bit_pos = max_bitwidth - bits_written;
            }
        }

        // Flush remaining bits
        if (bit_pos > 0) {
            out[out_ofs++] = current_word;
        }

        delete[] deltas;
    }

    block_offsets[num_blocks] = out_ofs;
    return out_ofs;
}

// GPU kernel for 64-bit delta decompression
__global__ void decompress_delta_64_kernel(
    const uint64_t* __restrict__ packed_data,
    uint64_t* __restrict__ output,
    const uint32_t* __restrict__ block_offsets,
    size_t num_blocks,
    size_t num_entries)
{
    int blk_idx = blockIdx.x;
    if (blk_idx >= num_blocks) return;

    int tid = threadIdx.x;
    size_t global_offset = blk_idx * BLOCK_SIZE_64;

    // Shared memory for prefix sum
    __shared__ int64_t shared_deltas[BLOCK_SIZE_64];
    __shared__ uint64_t s_first_val;
    __shared__ int64_t s_min_delta;
    __shared__ uint32_t s_bitwidth;

    // Get block data
    uint64_t block_offset = block_offsets[blk_idx];
    const uint64_t* block_data = packed_data + block_offset;

    // Thread 0 reads header
    if (tid == 0) {
        s_first_val = block_data[0];
        s_min_delta = (int64_t)block_data[1];
        s_bitwidth = block_data[2];
    }
    __syncthreads();

    uint64_t first_val = s_first_val;
    int64_t min_delta = s_min_delta;
    uint32_t bitwidth = s_bitwidth;
    const uint64_t* value_data = block_data + 3;

    uint64_t mask = (bitwidth == 64) ? ~0ULL : ((1ULL << bitwidth) - 1);

    // Each thread decodes multiple values
    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        int idx = tid + v * THREADS_PER_BLOCK;

        if (idx < BLOCK_SIZE_64 && global_offset + idx < num_entries) {
            // Calculate bit position
            uint64_t start_bit = (uint64_t)idx * bitwidth;
            uint64_t word_idx = start_bit / 64;
            int bit_offset = start_bit % 64;

            uint64_t val;
            if (bitwidth == 0) {
                val = 0;
            } else if (bit_offset + bitwidth <= 64) {
                val = (value_data[word_idx] >> bit_offset) & mask;
            } else {
                // Cross boundary
                uint64_t lo = value_data[word_idx] >> bit_offset;
                uint64_t hi = value_data[word_idx + 1] << (64 - bit_offset);
                val = (lo | hi) & mask;
            }

            shared_deltas[idx] = (int64_t)val + min_delta;
        }
    }
    __syncthreads();

    // Simple sequential prefix sum (for correctness, not performance)
    // In production, use CUB's block scan
    if (tid == 0) {
        int64_t running_sum = first_val;
        size_t block_size = min((size_t)BLOCK_SIZE_64, num_entries - global_offset);
        for (size_t i = 0; i < block_size; i++) {
            if (i == 0) {
                output[global_offset] = first_val;
            } else {
                running_sum += shared_deltas[i];
                output[global_offset + i] = (uint64_t)running_sum;
            }
        }
    }
}

template<typename T>
T* load_binary_file(const char* filename, size_t& count) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    size_t size = file.tellg();
    count = size / sizeof(T);
    file.seekg(0, std::ios::beg);

    T* data = new T[count];
    file.read(reinterpret_cast<char*>(data), size);
    file.close();

    return data;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <output_csv>" << std::endl;
        return 1;
    }

    cudaSetDevice(0);

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;

    const char* filename = argv[1];
    const char* output_csv = argv[2];

    // Load 64-bit data
    size_t num_entries = 0;
    uint64_t* original_data = load_binary_file<uint64_t>(filename, num_entries);

    if (!original_data) {
        std::cerr << "Failed to load data file" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << num_entries << " 64-bit values" << std::endl;

    // Align to block size
    size_t aligned_entries = ((num_entries + BLOCK_SIZE_64 - 1) / BLOCK_SIZE_64) * BLOCK_SIZE_64;
    if (aligned_entries != num_entries) {
        uint64_t* aligned_data = new uint64_t[aligned_entries];
        memcpy(aligned_data, original_data, num_entries * sizeof(uint64_t));
        for (size_t i = num_entries; i < aligned_entries; i++) {
            aligned_data[i] = original_data[num_entries - 1];
        }
        delete[] original_data;
        original_data = aligned_data;
    }

    size_t num_blocks = aligned_entries / BLOCK_SIZE_64;

    // Allocate compression buffers
    uint64_t* encoded_data = new uint64_t[aligned_entries * 2]();
    uint32_t* block_offsets = new uint32_t[num_blocks + 1]();

    // Measure compression time
    auto encode_start = std::chrono::high_resolution_clock::now();
    uint64_t encoded_size = delta_bin_pack_64(original_data, encoded_data, block_offsets, num_entries);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    std::cout << "Compression complete!" << std::endl;

    // Transfer to GPU
    uint64_t* d_encoded_data;
    uint32_t* d_block_offsets;
    uint64_t* d_output;

    cudaMalloc(&d_encoded_data, encoded_size * sizeof(uint64_t));
    cudaMalloc(&d_block_offsets, (num_blocks + 1) * sizeof(uint32_t));
    cudaMalloc(&d_output, aligned_entries * sizeof(uint64_t));

    cudaMemcpy(d_encoded_data, encoded_data, encoded_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_offsets, block_offsets, (num_blocks + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Warmup
    for (int i = 0; i < 3; i++) {
        decompress_delta_64_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_encoded_data, d_output, d_block_offsets, num_blocks, num_entries);
        cudaDeviceSynchronize();
    }

    // Measure decompression time
    const int num_trials = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start);
        decompress_delta_64_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_encoded_data, d_output, d_block_offsets, num_blocks, num_entries);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_time = 0.0f;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_time += trial_time;
    }

    float avg_decode_time_ms = total_time / num_trials;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify correctness
    uint64_t* temp = new uint64_t[aligned_entries];
    cudaMemcpy(temp, d_output, aligned_entries * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    bool correct = true;
    int error_count = 0;
    for (size_t i = 0; i < num_entries && error_count < 5; i++) {
        if (original_data[i] != temp[i]) {
            std::cout << "ERROR at " << i << ": expected " << original_data[i]
                      << " got " << temp[i] << std::endl;
            correct = false;
            error_count++;
        }
    }

    if (correct) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    // Calculate metrics
    size_t original_size = num_entries * sizeof(uint64_t);
    size_t compressed_size = encoded_size * sizeof(uint64_t);
    double compression_ratio = (double)original_size / (double)compressed_size;
    double encode_throughput_gbps = (original_size / (1024.0 * 1024.0 * 1024.0)) / (encode_time_ms / 1000.0);
    double decode_throughput_gbps = (original_size / (1024.0 * 1024.0 * 1024.0)) / (avg_decode_time_ms / 1000.0);

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Original Size: " << (original_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Compressed Size: " << (compressed_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Compression Ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "Encode Time: " << encode_time_ms << " ms" << std::endl;
    std::cout << "Decode Time: " << avg_decode_time_ms << " ms" << std::endl;
    std::cout << "Encode Throughput: " << encode_throughput_gbps << " GB/s" << std::endl;
    std::cout << "Decode Throughput: " << decode_throughput_gbps << " GB/s" << std::endl;

    // Write results to CSV
    std::ofstream csv_file;
    bool file_exists = std::ifstream(output_csv).good();
    csv_file.open(output_csv, std::ios::app);

    if (!file_exists) {
        csv_file << "algorithm,dataset,elements,data_size_mb,compressed_size_mb,compression_ratio,encode_time_ms,decode_time_ms,encode_throughput_gbps,decode_throughput_gbps,verification\n";
    }

    csv_file << "FastLanes-Delta-64,"
             << filename << ","
             << num_entries << ","
             << (original_size / (1024.0 * 1024.0)) << ","
             << (compressed_size / (1024.0 * 1024.0)) << ","
             << compression_ratio << ","
             << encode_time_ms << ","
             << avg_decode_time_ms << ","
             << encode_throughput_gbps << ","
             << decode_throughput_gbps << ","
             << (correct ? "PASS" : "FAIL") << "\n";

    csv_file.close();

    std::cout << "\nResults written to " << output_csv << std::endl;

    // Cleanup
    delete[] original_data;
    delete[] encoded_data;
    delete[] block_offsets;
    delete[] temp;
    cudaFree(d_encoded_data);
    cudaFree(d_block_offsets);
    cudaFree(d_output);

    return correct ? 0 : 1;
}
