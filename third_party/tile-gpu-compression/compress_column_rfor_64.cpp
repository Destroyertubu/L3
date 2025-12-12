// 使用GPU-RFOR (RLE + Frame of Reference) 格式压缩64位列数据
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

using namespace std;

// RLE编码并使用Frame of Reference压缩 (64-bit版本)
void rleBinPackEncode64(uint64_t* input, uint64_t* value_out, uint64_t* runlen_out,
                        uint32_t* val_offsets, uint32_t* rl_offsets, int num_entries) {
    const int tile_size = 512;

    uint32_t val_offset = 4; // 跳过header (64-bit words)
    uint32_t rl_offset = 4;

    // Header (4 x 64-bit words each)
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
        vector<uint64_t> values;
        vector<uint64_t> run_lengths;

        uint64_t current_val = input[tile_start];
        uint64_t run_len = 1;

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
        uint64_t val_min = values[0];
        uint64_t rl_min = run_lengths[0];
        for (int i = 1; i < rle_count; i++) {
            if (values[i] < val_min) val_min = values[i];
            if (run_lengths[i] < rl_min) rl_min = run_lengths[i];
        }

        // 计算bitwidth
        uint32_t val_bitwidth = 1;
        uint32_t rl_bitwidth = 1;
        for (int i = 0; i < rle_count; i++) {
            uint64_t val_delta = values[i] - val_min;
            uint64_t rl_delta = run_lengths[i] - rl_min;
            if (val_delta > 0) {
                uint32_t bw = 64 - __builtin_clzll(val_delta);
                if (bw > val_bitwidth) val_bitwidth = bw;
            }
            if (rl_delta > 0) {
                uint32_t bw = 64 - __builtin_clzll(rl_delta);
                if (bw > rl_bitwidth) rl_bitwidth = bw;
            }
        }

        // 存储value数据: reference (64-bit), bitwidth+padding (64-bit), count (64-bit)
        // Header layout matches 64-bit kernel expectations:
        // Word 0: reference (int64_t)
        // Word 1: bitwidth in lower 8 bits
        // Word 2: count
        value_out[val_offset++] = val_min;
        value_out[val_offset++] = val_bitwidth;  // bitwidth stored in lower bits
        value_out[val_offset++] = rle_count;

        // Pack values (64-bit aligned)
        uint32_t shift = 0;
        value_out[val_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint64_t delta = values[i] - val_min;

            if (shift + val_bitwidth > 64) {
                if (shift != 64) {
                    value_out[val_offset] |= (delta << shift);
                }
                val_offset++;
                value_out[val_offset] = 0;
                shift = (shift + val_bitwidth) & 63;
                if (shift > 0) {
                    value_out[val_offset] = delta >> (val_bitwidth - shift);
                }
            } else {
                value_out[val_offset] |= (delta << shift);
                shift += val_bitwidth;
            }
        }
        if (shift > 0) val_offset++;

        // 存储run_length数据
        runlen_out[rl_offset++] = rl_min;
        runlen_out[rl_offset++] = rl_bitwidth;
        runlen_out[rl_offset++] = rle_count;

        // Pack run_lengths (64-bit aligned)
        shift = 0;
        runlen_out[rl_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint64_t delta = run_lengths[i] - rl_min;

            if (shift + rl_bitwidth > 64) {
                if (shift != 64) {
                    runlen_out[rl_offset] |= (delta << shift);
                }
                rl_offset++;
                runlen_out[rl_offset] = 0;
                shift = (shift + rl_bitwidth) & 63;
                if (shift > 0) {
                    runlen_out[rl_offset] = delta >> (rl_bitwidth - shift);
                }
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
        cout << "Compresses 64-bit data using GPU-RFOR (RLE + Frame of Reference)" << endl;
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

    // 对齐到tile边界 (512)
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
    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    uint64_t* value_out = new uint64_t[num_entries * 2];
    uint64_t* runlen_out = new uint64_t[num_entries * 2];
    memset(value_out, 0, num_entries * 2 * sizeof(uint64_t));
    memset(runlen_out, 0, num_entries * 2 * sizeof(uint64_t));

    uint32_t* val_offsets = new uint32_t[num_tiles + 1];
    uint32_t* rl_offsets = new uint32_t[num_tiles + 1];

    cout << "Compressing 64-bit data with GPU-RFOR..." << endl;
    auto start = chrono::high_resolution_clock::now();
    rleBinPackEncode64(input, value_out, runlen_out, val_offsets, rl_offsets, num_entries);
    auto end = chrono::high_resolution_clock::now();
    double compression_time = chrono::duration<double, milli>(end - start).count();

    size_t val_size = val_offsets[num_tiles] * sizeof(uint64_t);
    size_t rl_size = rl_offsets[num_tiles] * sizeof(uint64_t);
    size_t total_compressed = val_size + rl_size;

    cout << "\nCompression complete!" << endl;
    cout << "  Compression time: " << compression_time << " ms" << endl;
    cout << "  Original size: " << num_entries * 8 << " bytes" << endl;
    cout << "  Compressed size (values): " << val_size << " bytes" << endl;
    cout << "  Compressed size (run_lengths): " << rl_size << " bytes" << endl;
    cout << "  Total compressed size: " << total_compressed << " bytes" << endl;
    cout << "  Compression ratio: " << (double)(num_entries * 8) / total_compressed << "x" << endl;

    // 保存压缩数据
    string val_file = output_prefix + "_rfor_64_val.bin";
    string rl_file = output_prefix + "_rfor_64_rl.bin";
    string val_off_file = output_prefix + "_rfor_64_val.binoff";
    string rl_off_file = output_prefix + "_rfor_64_rl.binoff";

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
