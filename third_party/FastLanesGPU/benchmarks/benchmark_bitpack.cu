#include "config.hpp"
#include "cub/util_debug.cuh"
#include "kernel.cuh"
#include "binpack_kernel.cuh"
#include <cuda_profiler_api.h>
#include <fstream>
#include <iostream>
#include <chrono>

uint32_t bin_pack(uint32_t*& in, uint32_t*& out, uint32_t*& block_offsets, uint32_t tup_c) {
	uint32_t out_ofs = 0;

	uint32_t block_size      = 128;
	uint32_t miniblock_count = 4;
	uint32_t miniblock_size  = block_size / miniblock_count;
	uint32_t total_count     = tup_c;
	uint32_t first_val       = in[0];

	out[0] = block_size;
	out[1] = miniblock_count;
	out[2] = total_count;
	out[3] = first_val;

	out_ofs += 4;

	for (uint32_t idx = 0; idx < tup_c; idx += block_size) {
		uint32_t blk_idx       = idx / block_size;
		block_offsets[blk_idx] = out_ofs;

		// Find min val
		uint32_t min_val = in[0];
		for (int i = 1; i < block_size; i++) {
			if (in[i] < min_val) { min_val = in[i]; }
		}

		for (int i = 0; i < block_size; i++) {
			in[i] = in[i] - min_val;
		}

		uint32_t* miniblock_bitwidths = new uint32_t[miniblock_count];
		for (int i = 0; i < miniblock_count; i++) {
			miniblock_bitwidths[i] = 0;
		}

		for (uint32_t miniblock = 0; miniblock < miniblock_count; miniblock++) {
			for (uint32_t i = 0; i < miniblock_size; i++) {
				uint32_t bitwidth = uint32_t(ceil(log2(in[miniblock * miniblock_size + i] + 1)));
				if (bitwidth > miniblock_bitwidths[miniblock]) { miniblock_bitwidths[miniblock] = bitwidth; }
			}
		}

		// Extra for Simple BinPack
		uint32_t max_bitwidth = miniblock_bitwidths[0];
		for (int i = 1; i < miniblock_count; i++) {
			max_bitwidth = std::max(max_bitwidth, miniblock_bitwidths[i]);
		}
		for (int i = 0; i < miniblock_count; i++) {
			miniblock_bitwidths[i] = max_bitwidth;
		}

		out[out_ofs] = min_val;
		out_ofs++;

		out[out_ofs] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) + (miniblock_bitwidths[2] << 16) +
		               (miniblock_bitwidths[3] << 24);
		out_ofs++;

		for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
			uint32_t bitwidth = miniblock_bitwidths[miniblock];
			uint32_t shift    = 0;
			for (int i = 0; i < miniblock_size; i++) {
				if (shift + bitwidth > 32) {
					if (shift != 32) { out[out_ofs] += in[miniblock * miniblock_size + i] << shift; }
					out_ofs++;
					shift        = (shift + bitwidth) & (32 - 1);
					out[out_ofs] = in[miniblock * miniblock_size + i] >> (bitwidth - shift);
				} else {
					out[out_ofs] += in[miniblock * miniblock_size + i] << shift;
					shift += bitwidth;
				}
			}
			out_ofs++;
		}

		// Increment the input pointer by block size
		in += block_size;
	}

	block_offsets[tup_c / block_size] = out_ofs;

	return out_ofs;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void run_bin_kernel(int* col, uint* col_block_start, uint* col_data) {

	int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
	int tile_idx  = blockIdx.x;

	// Load a segment of consecutive items that are blocked across threads
	uint32_t col_block[ITEMS_PER_THREAD];

	extern __shared__ uint shared_buffer[];

	LoadBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, shared_buffer, col_block);

	// write unpacked values directly to global memory
	for (int i = 0; i < ITEMS_PER_THREAD; i++) {
		col[tile_size * tile_idx + i * 128 + threadIdx.x] = col_block[i];
	}
}

namespace tile_based {
template <typename T>
T* loadColumnToGPU(T* src, int len) {
	T* dest = nullptr;
	cudaMalloc((void**)&dest, sizeof(T) * len);
	CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * len, cudaMemcpyHostToDevice));
	return dest;
}
} // namespace tile_based

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

	const char* filename = argv[1];
	const char* output_csv = argv[2];

	// Load data
	size_t tup_c = 0;
	uint32_t* original_data = load_binary_file<uint32_t>(filename, tup_c);

	if (!original_data) {
		std::cerr << "Failed to load data file" << std::endl;
		return 1;
	}

	// Ensure data size is multiple of 128
	size_t aligned_tup_c = ((tup_c + 127) / 128) * 128;
	if (aligned_tup_c != tup_c) {
		uint32_t* aligned_data = new uint32_t[aligned_tup_c];
		memcpy(aligned_data, original_data, tup_c * sizeof(uint32_t));
		for (size_t i = tup_c; i < aligned_tup_c; i++) {
			aligned_data[i] = original_data[tup_c - 1];
		}
		delete[] original_data;
		original_data = aligned_data;
		tup_c = aligned_tup_c;
	}

	int      block_size      = 128;
	int      elem_per_thread = 4;
	int      tile_size       = block_size * elem_per_thread;
	int      num_blocks      = tup_c / block_size;
	auto*    encoded_data    = new uint32_t[tup_c * 2](); // Allocate more space for safety
	uint64_t ofs_c           = num_blocks + 1;
	auto*    ofs_arr         = new uint32_t[ofs_c]();
	auto*    copy_data       = new uint32_t[tup_c];

	/* Data needs to be copied. the encoding changes the original data. */
	memcpy(copy_data, original_data, tup_c * sizeof(uint32_t));

	// Measure compression time
	auto encode_start = std::chrono::high_resolution_clock::now();
	uint32_t encoded_data_bsz = bin_pack(copy_data, encoded_data, ofs_arr, tup_c);
	auto encode_end = std::chrono::high_resolution_clock::now();
	double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

	tile_based::encoded_column h_col {ofs_arr, encoded_data, tup_c * 4};

	uint* d_col_block_start = tile_based::loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1);
	uint* d_col_data        = tile_based::loadColumnToGPU<uint>(h_col.data, encoded_data_bsz);

	tile_based::encoded_column d_col {d_col_block_start, d_col_data};

	cudaDeviceSynchronize();

	const int num_threads      = 128;
	const int items_per_thread = 4;
	int*      col              = nullptr;
	cudaMalloc((void**)&col, tup_c * sizeof(int));
	size_t Dg = (tup_c + tile_size - 1) / tile_size;
	size_t Db = num_threads;
	size_t Ns = 3000;

	// Warmup
	for (int i = 0; i < 3; i++) {
		run_bin_kernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(col, d_col.block_start, d_col.data);
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
		run_bin_kernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(col, d_col.block_start, d_col.data);
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
	int* temp = new int[tup_c];
	CubDebugExit(cudaMemcpy(temp, col, sizeof(int) * tup_c, cudaMemcpyDeviceToHost));

	bool correct = true;
	for (size_t i = 0; i < tup_c; i++) {
		if (original_data[i] != (uint32_t)temp[i]) {
			std::cout << "ERROR:" << i << " " << original_data[i] << " " << temp[i] << '\n';
			correct = false;
			break;
		}
	}

	if (correct) {
		std::cout << "Verification passed!" << std::endl;
	}

	// Calculate metrics
	size_t original_size = tup_c * sizeof(uint32_t);
	size_t compressed_size = encoded_data_bsz * sizeof(uint32_t);
	double compression_ratio = (double)original_size / (double)compressed_size;
	double encode_throughput_gbps = (original_size / (1024.0 * 1024.0 * 1024.0)) / (encode_time_ms / 1000.0);
	double decode_throughput_gbps = (original_size / (1024.0 * 1024.0 * 1024.0)) / (avg_decode_time_ms / 1000.0);

	// Write results to CSV
	std::ofstream csv_file;
	bool file_exists = std::ifstream(output_csv).good();
	csv_file.open(output_csv, std::ios::app);

	if (!file_exists) {
		csv_file << "algorithm,dataset,data_size_mb,compressed_size_mb,compression_ratio,encode_time_ms,decode_time_ms,encode_throughput_gbps,decode_throughput_gbps\n";
	}

	csv_file << "BitPack,"
	         << filename << ","
	         << (original_size / (1024.0 * 1024.0)) << ","
	         << (compressed_size / (1024.0 * 1024.0)) << ","
	         << compression_ratio << ","
	         << encode_time_ms << ","
	         << avg_decode_time_ms << ","
	         << encode_throughput_gbps << ","
	         << decode_throughput_gbps << "\n";

	csv_file.close();

	std::cout << "Results written to " << output_csv << std::endl;
	std::cout << "Compression Ratio: " << compression_ratio << std::endl;
	std::cout << "Encode Time: " << encode_time_ms << " ms" << std::endl;
	std::cout << "Decode Time: " << avg_decode_time_ms << " ms" << std::endl;
	std::cout << "Decode Throughput: " << decode_throughput_gbps << " GB/s" << std::endl;

	// Cleanup
	delete[] original_data;
	delete[] encoded_data;
	delete[] ofs_arr;
	delete[] copy_data;
	delete[] temp;
	cudaFree(col);
	cudaFree(d_col_block_start);
	cudaFree(d_col_data);

	return 0;
}
