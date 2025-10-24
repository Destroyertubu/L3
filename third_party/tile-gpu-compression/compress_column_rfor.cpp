// 使用GPU-RFOR (RLE + Frame of Reference) 格式压缩列数据
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

using namespace std;

// RLE编码并使用Frame of Reference压缩
void rleBinPackEncode(uint32_t* input, uint32_t* value_out, uint32_t* runlen_out,
                      uint32_t* val_offsets, uint32_t* rl_offsets, int num_entries) {
    const int tile_size = 512;

    uint32_t val_offset = 4; // 跳过header
    uint32_t rl_offset = 4;

    // Header
    value_out[0] = tile_size;
    value_out[1] = 1; // miniblock_count (simplified)
    value_out[2] = num_entries;
    value_out[3] = input[0];

    runlen_out[0] = tile_size;
    runlen_out[1] = 1;
    runlen_out[2] = num_entries;
    runlen_out[3] = input[0];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_size;
        int tile_end = min(tile_start + tile_size, num_entries);
        int tile_len = tile_end - tile_start;

        val_offsets[tile_idx] = val_offset;
        rl_offsets[tile_idx] = rl_offset;

        // RLE编码
        vector<uint32_t> values;
        vector<uint32_t> run_lengths;

        uint32_t current_val = input[tile_start];
        uint32_t run_len = 1;

        for (int i = tile_start + 1; i < tile_end; i++) {
            if (input[i] == current_val) {
                run_len++;
            } else {
                values.push_back(current_val);
                run_lengths.push_back(run_len);
                current_val = input[i];
                run_len = 1;
            }
        }
        // 最后一个run
        values.push_back(current_val);
        run_lengths.push_back(run_len);

        int rle_count = values.size();

        // 找最小值作为reference
        uint32_t val_min = values[0];
        uint32_t rl_min = run_lengths[0];
        for (int i = 1; i < rle_count; i++) {
            if (values[i] < val_min) val_min = values[i];
            if (run_lengths[i] < rl_min) rl_min = run_lengths[i];
        }

        // 计算bitwidth
        uint32_t val_bitwidth = 1;
        uint32_t rl_bitwidth = 1;
        for (int i = 0; i < rle_count; i++) {
            uint32_t val_delta = values[i] - val_min;
            uint32_t rl_delta = run_lengths[i] - rl_min;
            if (val_delta > 0) {
                uint32_t bw = 32 - __builtin_clz(val_delta);
                if (bw > val_bitwidth) val_bitwidth = bw;
            }
            if (rl_delta > 0) {
                uint32_t bw = 32 - __builtin_clz(rl_delta);
                if (bw > rl_bitwidth) rl_bitwidth = bw;
            }
        }

        // 存储value数据: reference, bitwidth, count
        value_out[val_offset++] = val_min;
        value_out[val_offset++] = val_bitwidth;
        value_out[val_offset++] = rle_count;

        // Pack values
        uint32_t shift = 0;
        value_out[val_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint32_t delta = values[i] - val_min;

            if (shift + val_bitwidth > 32) {
                if (shift != 32) {
                    value_out[val_offset] |= (delta << shift);
                }
                val_offset++;
                value_out[val_offset] = 0;
                shift = (shift + val_bitwidth) & 31;
                value_out[val_offset] = delta >> (val_bitwidth - shift);
            } else {
                value_out[val_offset] |= (delta << shift);
                shift += val_bitwidth;
            }
        }
        if (shift > 0) val_offset++;

        // 存储run_length数据: reference, bitwidth, count
        runlen_out[rl_offset++] = rl_min;
        runlen_out[rl_offset++] = rl_bitwidth;
        runlen_out[rl_offset++] = rle_count;

        // Pack run_lengths
        shift = 0;
        runlen_out[rl_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint32_t delta = run_lengths[i] - rl_min;

            if (shift + rl_bitwidth > 32) {
                if (shift != 32) {
                    runlen_out[rl_offset] |= (delta << shift);
                }
                rl_offset++;
                runlen_out[rl_offset] = 0;
                shift = (shift + rl_bitwidth) & 31;
                runlen_out[rl_offset] = delta >> (rl_bitwidth - shift);
            } else {
                runlen_out[rl_offset] |= (delta << shift);
                shift += rl_bitwidth;
            }
        }
        if (shift > 0) rl_offset++;
    }

    val_offsets[num_tiles] = val_offset;
    rl_offsets[num_tiles] = rl_offset;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input.bin> <output_prefix>" << endl;
        cout << "Compresses using GPU-RFOR (RLE + Frame of Reference)" << endl;
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

    int num_entries = file_size / sizeof(uint32_t);
    cout << "Loading " << num_entries << " values..." << endl;

    uint32_t* input = new uint32_t[num_entries];
    infile.read(reinterpret_cast<char*>(input), file_size);
    infile.close();

    // 对齐到tile边界 (512)
    const int tile_size = 512;
    int adjusted_len = ((num_entries + tile_size - 1) / tile_size) * tile_size;
    if (adjusted_len > num_entries) {
        uint32_t* aligned_input = new uint32_t[adjusted_len];
        memcpy(aligned_input, input, num_entries * sizeof(uint32_t));
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
    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    uint32_t* value_out = new uint32_t[num_entries * 2];
    uint32_t* runlen_out = new uint32_t[num_entries * 2];
    memset(value_out, 0, num_entries * 2 * sizeof(uint32_t));
    memset(runlen_out, 0, num_entries * 2 * sizeof(uint32_t));

    uint32_t* val_offsets = new uint32_t[num_tiles + 1];
    uint32_t* rl_offsets = new uint32_t[num_tiles + 1];

    cout << "Compressing with GPU-RFOR..." << endl;
    auto start = chrono::high_resolution_clock::now();
    rleBinPackEncode(input, value_out, runlen_out, val_offsets, rl_offsets, num_entries);
    auto end = chrono::high_resolution_clock::now();
    double compression_time = chrono::duration<double, milli>(end - start).count();

    size_t val_size = val_offsets[num_tiles] * sizeof(uint32_t);
    size_t rl_size = rl_offsets[num_tiles] * sizeof(uint32_t);
    size_t total_compressed = val_size + rl_size;

    cout << "\nCompression complete!" << endl;
    cout << "  Compression time: " << compression_time << " ms" << endl;
    cout << "  Original size: " << num_entries * 4 << " bytes" << endl;
    cout << "  Compressed size (values): " << val_size << " bytes" << endl;
    cout << "  Compressed size (run_lengths): " << rl_size << " bytes" << endl;
    cout << "  Total compressed size: " << total_compressed << " bytes" << endl;
    cout << "  Compression ratio: " << (double)(num_entries * 4) / total_compressed << "x" << endl;

    // 保存压缩数据
    string val_file = output_prefix + "_val.bin";
    string rl_file = output_prefix + "_rl.bin";
    string val_off_file = output_prefix + "_val.binoff";
    string rl_off_file = output_prefix + "_rl.binoff";

    ofstream outval(val_file, ios::binary);
    outval.write(reinterpret_cast<char*>(value_out), val_size);
    outval.close();

    ofstream outrl(rl_file, ios::binary);
    outrl.write(reinterpret_cast<char*>(runlen_out), rl_size);
    outrl.close();

    ofstream outvaloff(val_off_file, ios::binary);
    outvaloff.write(reinterpret_cast<char*>(val_offsets), (num_tiles + 1) * sizeof(uint32_t));
    outvaloff.close();

    ofstream outrloff(rl_off_file, ios::binary);
    outrloff.write(reinterpret_cast<char*>(rl_offsets), (num_tiles + 1) * sizeof(uint32_t));
    outrloff.close();

    cout << "\nOutput files:" << endl;
    cout << "  Values: " << val_file << endl;
    cout << "  Run lengths: " << rl_file << endl;
    cout << "  Value offsets: " << val_off_file << endl;
    cout << "  RL offsets: " << rl_off_file << endl;

    delete[] input;
    delete[] value_out;
    delete[] runlen_out;
    delete[] val_offsets;
    delete[] rl_offsets;

    return 0;
}
