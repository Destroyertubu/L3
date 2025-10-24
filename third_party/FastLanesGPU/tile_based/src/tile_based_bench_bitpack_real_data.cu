#include "config.hpp"
#include "cub/util_debug.cuh"
#include "kernel.cuh"
#include "binpack_kernel.cuh"
#include "utils/gpu_utils.h"
#include <cuda_profiler_api.h>
#include <chrono>
#include <fstream>
#include <cstring>

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

uint64_t load_data_from_file(const char* filename, uint32_t** data, uint64_t max_elements) {
	std::string fname(filename);
	uint64_t count = 0;

	if (fname.find(".bin") != std::string::npos) {
		// Binary file
		FILE* f = fopen(filename, "rb");
		if (!f) {
			std::cerr << "Cannot open file: " << filename << std::endl;
			return 0;
		}

		fseek(f, 0, SEEK_END);
		long file_size = ftell(f);
		fseek(f, 0, SEEK_SET);

		// Check if uint32 or uint64
		if (fname.find("uint32") != std::string::npos) {
			count = std::min((uint64_t)(file_size / sizeof(uint32_t)), max_elements);
			*data = new uint32_t[count];
			fread(*data, sizeof(uint32_t), count, f);
		} else {
			// uint64 - convert to uint32
			count = std::min((uint64_t)(file_size / sizeof(uint64_t)), max_elements);
			uint64_t* temp = new uint64_t[count];
			fread(temp, sizeof(uint64_t), count, f);
			*data = new uint32_t[count];
			for (uint64_t i = 0; i < count; i++) {
				(*data)[i] = (uint32_t)temp[i];
			}
			delete[] temp;
		}
		fclose(f);
	} else {
		// Text file
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Cannot open file: " << filename << std::endl;
			return 0;
		}

		std::vector<uint32_t> temp_data;
		uint64_t value;
		while (file >> value && temp_data.size() < max_elements) {
			temp_data.push_back((uint32_t)value);
		}
		file.close();

		count = temp_data.size();
		*data = new uint32_t[count];
		memcpy(*data, temp_data.data(), count * sizeof(uint32_t));
	}

	return count;
}

int main(int argc, char** argv) {

	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <data_file> <bitwidth> [dataset_name]" << std::endl;
		return 1;
	}

	cudaSetDevice(0);

	const char* data_file = argv[1];
	int bitwidth = atoi(argv[2]);
	const char* dataset_name = (argc > 3) ? argv[3] : "unknown";

	std::cout << "Dataset: " << dataset_name << std::endl;
	std::cout << "Bitwidth: " << bitwidth << std::endl;
	std::cout << "Loading data from: " << data_file << std::endl;

	uint32_t* original_data = nullptr;
	uint64_t tup_c = load_data_from_file(data_file, &original_data, 1 << 28);  // Max 256M elements

	if (tup_c == 0) {
		std::cerr << "Failed to load data" << std::endl;
		return 1;
	}

	// Round down to multiple of 128
	tup_c = (tup_c / 128) * 128;

	std::cout << "Loaded " << tup_c << " elements (" << (tup_c * sizeof(uint32_t)) / (1024.0 * 1024.0) << " MB)" << std::endl;

	int      block_size      = 128;
	int      elem_per_thread = 4;
	int      tile_size       = block_size * elem_per_thread;
	int      num_blocks      = tup_c / block_size;
	auto*    encoded_data    = new uint32_t[tup_c * 2]();  // Extra space for safety
	uint64_t ofs_c           = num_blocks + 1;
	auto*    ofs_arr         = new uint32_t[ofs_c]();
	auto*    copy_data       = new uint32_t[tup_c];

	/* Data needs to be copied. the encoding changes the original data. */
	memcpy(copy_data, original_data, tup_c * sizeof(uint32_t));

	// Measure encoding time
	auto encode_start = std::chrono::high_resolution_clock::now();
	uint32_t encoded_data_bsz = bin_pack(copy_data, encoded_data, ofs_arr, tup_c);
	auto encode_end = std::chrono::high_resolution_clock::now();
	double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

	tile_based::encoded_column h_col {ofs_arr, encoded_data, tup_c * 4};

	// Measure H2D transfer time
	cudaEvent_t h2d_start, h2d_stop;
	cudaEventCreate(&h2d_start);
	cudaEventCreate(&h2d_stop);

	cudaEventRecord(h2d_start);
	uint* d_col_block_start = tile_based::loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1);
	uint* d_col_data        = tile_based::loadColumnToGPU<uint>(h_col.data, encoded_data_bsz);
	cudaEventRecord(h2d_stop);
	cudaEventSynchronize(h2d_stop);

	float h2d_time_ms;
	cudaEventElapsedTime(&h2d_time_ms, h2d_start, h2d_stop);

	tile_based::encoded_column d_col {d_col_block_start, d_col_data};

	cudaDeviceSynchronize();

	const int num_threads      = 128;
	const int items_per_thread = 4;
	int*      col              = nullptr;
	cudaMalloc((void**)&col, tup_c * sizeof(int));
	size_t Dg = (tup_c + tile_size - 1) / tile_size;
	size_t Db = num_threads;
	size_t Ns = 3000;

	run_bin_kernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(col, d_col.block_start, d_col.data);

	int* temp = new int[tup_c];

	// Measure D2H transfer time
	cudaEvent_t d2h_start, d2h_stop;
	cudaEventCreate(&d2h_start);
	cudaEventCreate(&d2h_stop);

	cudaEventRecord(d2h_start);
	CubDebugExit(cudaMemcpy(temp, col, sizeof(int) * tup_c, cudaMemcpyDeviceToHost));
	cudaEventRecord(d2h_stop);
	cudaEventSynchronize(d2h_stop);

	float d2h_time_ms;
	cudaEventElapsedTime(&d2h_time_ms, d2h_start, d2h_stop);

	// Verify correctness
	bool correct = true;
	for (uint64_t i = 0; i < tup_c; i++) {
		if (original_data[i] != (uint32_t)temp[i]) {
			std::cout << "ERROR at " << i << ": " << original_data[i] << " != " << temp[i] << '\n';
			correct = false;
			break;
		}
	}
	if (correct) {
		std::cout << "✓ Verification passed!" << '\n';
	}

	// Run decode trials
	int num_trials = 10;
	float query_time;
	SETUP_TIMING();

	// Warmup
	for (int t = 0; t < 3; t++) {
		run_bin_kernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(col, d_col.block_start, d_col.data);
		cudaDeviceSynchronize();
	}

	float total_decode_time = 0.0f;
	for (int t = 0; t < num_trials; t++) {
		cudaEventRecord(start, nullptr);
		run_bin_kernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(col, d_col.block_start, d_col.data);
		cudaEventRecord(stop, nullptr);

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&query_time, start, stop);
		total_decode_time += query_time;

		CubDebugExit(cudaPeekAtLastError());
		CubDebugExit(cudaDeviceSynchronize());
	}

	float avg_decode_time = total_decode_time / num_trials;

	// Calculate metrics
	size_t original_size_bytes = tup_c * sizeof(uint32_t);
	size_t compressed_size_bytes = encoded_data_bsz * sizeof(uint32_t);
	double compression_ratio = (double)original_size_bytes / (double)compressed_size_bytes;
	double original_size_mb = original_size_bytes / (1024.0 * 1024.0);
	double compressed_size_mb = compressed_size_bytes / (1024.0 * 1024.0);
	double decode_throughput_gbps = (original_size_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_decode_time / 1000.0);

	// Output JSON format for easy parsing
	std::cout << "\n========== BENCHMARK RESULTS JSON ==========" << std::endl;
	std::cout << "{" << std::endl;
	std::cout << "  \"dataset\": \"" << dataset_name << "\"," << std::endl;
	std::cout << "  \"algorithm\": \"BitPack-" << bitwidth << "bit\"," << std::endl;
	std::cout << "  \"bitwidth\": " << bitwidth << "," << std::endl;
	std::cout << "  \"data_size_mb\": " << original_size_mb << "," << std::endl;
	std::cout << "  \"compressed_size_mb\": " << compressed_size_mb << "," << std::endl;
	std::cout << "  \"compression_ratio\": " << compression_ratio << "," << std::endl;
	std::cout << "  \"encode_time_ms\": " << encode_time_ms << "," << std::endl;
	std::cout << "  \"h2d_transfer_ms\": " << h2d_time_ms << "," << std::endl;
	std::cout << "  \"decode_time_ms\": " << avg_decode_time << "," << std::endl;
	std::cout << "  \"d2h_transfer_ms\": " << d2h_time_ms << "," << std::endl;
	std::cout << "  \"decode_throughput_gbps\": " << decode_throughput_gbps << std::endl;
	std::cout << "}" << std::endl;
	std::cout << "===========================================" << std::endl;

	// Cleanup
	delete[] original_data;
	delete[] encoded_data;
	delete[] ofs_arr;
	delete[] copy_data;
	delete[] temp;
	cudaFree(col);
	cudaFree(d_col_block_start);
	cudaFree(d_col_data);
}
