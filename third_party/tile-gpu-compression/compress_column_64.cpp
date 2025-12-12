// 使用GPU-FOR格式压缩64位列数据
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <chrono>

using namespace std;

void binPackEncode64(uint64_t* input, uint64_t* output, uint32_t* block_offsets, int num_entries) {
    uint32_t offset = 4; // 跳过header (64-bit words)
    const int block_size = 128;
    const int miniblock_count = 4;
    const int miniblock_size = 32;

    // Header (4 x 64-bit words)
    output[0] = block_size;
    output[1] = miniblock_count;
    output[2] = num_entries;
    output[3] = input[0];

    int num_blocks = (num_entries + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        block_offsets[b] = offset;
        int block_start = b * block_size;
        int block_end = min(block_start + block_size, num_entries);
        int actual_size = block_end - block_start;

        // 找最小值 (reference)
        uint64_t min_val = input[block_start];
        for (int i = 1; i < actual_size; i++) {
            if (input[block_start + i] < min_val)
                min_val = input[block_start + i];
        }

        // 存储reference (64-bit)
        output[offset++] = min_val;

        // 计算bitwidths (存储在一个64-bit word中, 每个miniblock 8位, 最大64)
        uint32_t bitwidths[4] = {0, 0, 0, 0};
        for (int m = 0; m < miniblock_count; m++) {
            int mb_start = block_start + m * miniblock_size;
            int mb_end = min(mb_start + miniblock_size, num_entries);

            for (int i = mb_start; i < mb_end; i++) {
                uint64_t delta = input[i] - min_val;
                if (delta > 0) {
                    uint32_t bw = 64 - __builtin_clzll(delta);
                    if (bw > bitwidths[m]) bitwidths[m] = bw;
                }
            }
            if (bitwidths[m] == 0) bitwidths[m] = 1;
        }

        // 存储bitwidths (packed into 64-bit word)
        uint64_t packed_bw = (uint64_t)bitwidths[0] | ((uint64_t)bitwidths[1] << 8) |
                             ((uint64_t)bitwidths[2] << 16) | ((uint64_t)bitwidths[3] << 24);
        output[offset++] = packed_bw;

        // Pack每个miniblock (64-bit aligned)
        for (int m = 0; m < miniblock_count; m++) {
            int mb_start = block_start + m * miniblock_size;
            int mb_end = min(mb_start + miniblock_size, num_entries);
            uint32_t bitwidth = bitwidths[m];

            uint32_t shift = 0;
            output[offset] = 0;

            for (int i = mb_start; i < mb_end; i++) {
                uint64_t delta = input[i] - min_val;

                if (shift + bitwidth > 64) {
                    if (shift != 64) {
                        output[offset] |= (delta << shift);
                    }
                    offset++;
                    output[offset] = 0;
                    shift = (shift + bitwidth) & 63;
                    if (shift > 0) {
                        output[offset] = delta >> (bitwidth - shift);
                    }
                } else {
                    output[offset] |= (delta << shift);
                    shift += bitwidth;
                }
            }

            // 补齐到64位边界
            if (shift > 0) {
                offset++;
                output[offset] = 0;
            }
        }
    }

    block_offsets[num_blocks] = offset;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input.bin> <output_prefix>" << endl;
        cout << "Compresses 64-bit data using GPU-FOR (Frame of Reference)" << endl;
        return 1;
    }

    string input_file = argv[1];
    string output_prefix = argv[2];

    // 读取二进制数据
    ifstream infile(input_file, ios::binary);
    if (!infile) {
        cerr << "Error: Cannot open input file " << input_file << endl;
        return 1;
    }

    infile.seekg(0, ios::end);
    size_t file_size = infile.tellg();
    infile.seekg(0, ios::beg);

    int num_entries = file_size / sizeof(uint64_t);
    cout << "Loading " << num_entries << " 64-bit values..." << endl;

    uint64_t* input = new uint64_t[num_entries];
    infile.read(reinterpret_cast<char*>(input), file_size);
    infile.close();

    // 对齐到tile边界
    const int tile_size = 512;
    int adjusted_len = ((num_entries + tile_size - 1) / tile_size) * tile_size;
    if (adjusted_len > num_entries) {
        uint64_t* aligned_input = new uint64_t[adjusted_len];
        memcpy(aligned_input, input, num_entries * sizeof(uint64_t));
        // 用最后一个值填充
        for (int i = num_entries; i < adjusted_len; i++) {
            aligned_input[i] = input[num_entries - 1];
        }
        delete[] input;
        input = aligned_input;
        num_entries = adjusted_len;
    }

    cout << "Adjusted length: " << num_entries << " (aligned to tile boundary)" << endl;

    // 压缩
    int num_blocks = num_entries / 128;
    uint64_t* output = new uint64_t[num_entries * 2]; // 预留足够空间
    memset(output, 0, num_entries * 2 * sizeof(uint64_t));
    uint32_t* offsets = new uint32_t[num_blocks + 1];

    cout << "Compressing 64-bit data with GPU-FOR..." << endl;
    auto start = chrono::high_resolution_clock::now();
    binPackEncode64(input, output, offsets, num_entries);
    auto end = chrono::high_resolution_clock::now();
    double compression_time = chrono::duration<double, milli>(end - start).count();

    size_t compressed_size = offsets[num_blocks] * sizeof(uint64_t);
    cout << "\nCompression complete!" << endl;
    cout << "  Compression time: " << compression_time << " ms" << endl;
    cout << "  Original size: " << num_entries * 8 << " bytes" << endl;
    cout << "  Compressed size: " << compressed_size << " bytes" << endl;
    cout << "  Compression ratio: " << (double)(num_entries * 8) / compressed_size << "x" << endl;

    // 保存压缩数据
    string data_file = output_prefix + "_64.bin";
    string offset_file = output_prefix + "_64.binoff";

    ofstream outdata(data_file, ios::binary);
    outdata.write(reinterpret_cast<char*>(output), compressed_size);
    outdata.close();

    ofstream outoffset(offset_file, ios::binary);
    outoffset.write(reinterpret_cast<char*>(offsets), (num_blocks + 1) * sizeof(uint32_t));
    outoffset.close();

    cout << "\nOutput files:" << endl;
    cout << "  Data: " << data_file << endl;
    cout << "  Offsets: " << offset_file << endl;

    delete[] input;
    delete[] output;
    delete[] offsets;

    return 0;
}
