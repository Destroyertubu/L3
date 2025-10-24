#include "config.hpp"
#include "cub/util_debug.cuh"
#include "kernel.cuh"
#include "binpack_kernel.cuh"
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <fstream>
#include <iostream>
#include <chrono>

uint deltaBinPack(int*& in, int*& out, uint*& block_offsets, uint num_entries) {
	uint offset = 0;

	uint block_size      = 128;
	uint elem_per_thread = 4;
	uint tile_size       = block_size * elem_per_thread;

	uint miniblock_count = 4;
	uint total_count     = num_entries;
	uint first_val       = in[0];

	out[0] = block_size;
	out[1] = miniblock_count;
	out[2] = total_count;
	out[3] = first_val;

	offset += 4;

	for (uint tile_start = 0; tile_start < num_entries; tile_start += tile_size) {
		uint block_index   = tile_start / block_size;
		int  tmp_first_val = in[0];

		out[offset] = tmp_first_val;
		offset++;

		// Compute the deltas
		for (int i = tile_size - 1; i > 0; i--) {
			in[i] = in[i] - in[i - 1];
		}
		in[0] = 0;

		for (int block_start = 0; block_start < block_size * 4; block_start += block_size, block_index += 1) {
			block_offsets[block_index] = offset;

			// For FOR - Find min val
			int min_val = in[0];
			for (int i = 1; i < block_size; i++) {
				if (in[i] < min_val) { min_val = in[i]; }
			}

			min_val = 0; /* HACK */
			for (int i = 0; i < block_size; i++) {
				in[i] = in[i] - min_val;
			}

			out[offset] = min_val;
			offset++;

			// Subtracting min_val ensures that all input vals are >= 0
			// Going forward in and out will both be treated as unsigned integers.
			uint* inp  = (uint*)in;
			uint* outp = (uint*)out;

			uint  miniblock_size      = block_size / miniblock_count;
			uint* miniblock_bitwidths = new uint[miniblock_count];
			for (int i = 0; i < miniblock_count; i++) {
				miniblock_bitwidths[i] = 0;
			}

			for (uint miniblock = 0; miniblock < miniblock_count; miniblock++) {
				for (uint i = 0; i < miniblock_size; i++) {
					uint bitwidth = uint(ceil(log2(inp[miniblock * miniblock_size + i] + 1)));
					if (bitwidth > miniblock_bitwidths[miniblock]) { miniblock_bitwidths[miniblock] = bitwidth; }
				}
			}

			// Extra for Simple BinPack
			uint max_bitwidth = miniblock_bitwidths[0];
			for (int i = 1; i < miniblock_count; i++) {
				max_bitwidth = max(max_bitwidth, miniblock_bitwidths[i]);
			}
			for (int i = 0; i < miniblock_count; i++) {
				miniblock_bitwidths[i] = max_bitwidth;
			}
			outp[offset] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) + (miniblock_bitwidths[2] << 16) +
			               (miniblock_bitwidths[3] << 24);
			offset++;

			for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
				uint bitwidth = miniblock_bitwidths[miniblock];
				uint shift    = 0;
				for (int i = 0; i < miniblock_size; i++) {
					if (shift + bitwidth > 32) {
						if (shift != 32) { outp[offset] += inp[miniblock * miniblock_size + i] << shift; }
						offset++;
						shift        = (shift + bitwidth) & (32 - 1);
						outp[offset] = inp[miniblock * miniblock_size + i] >> (bitwidth - shift);
					} else {
						outp[offset] += inp[miniblock * miniblock_size + i] << shift;
						shift += bitwidth;
					}
				}
				offset++;
			}

			// Increment the input pointer by block size
			in += block_size;
		}
	}

	block_offsets[num_entries / block_size] = offset;

	return offset;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runDBinKernel(int* col, uint* col_block_start, uint* col_data, int num_entries) {
	int tile_size   = BLOCK_THREADS * ITEMS_PER_THREAD;
	int tile_idx    = blockIdx.x;
	int tile_offset = tile_idx * tile_size;

	// Load a segment of consecutive items that are blocked across threads
	int col_block[ITEMS_PER_THREAD];

	int  num_tiles      = (num_entries + tile_size - 1) / tile_size;
	int  num_tile_items = tile_size;
	bool is_last_tile   = false;
	if (tile_idx == num_tiles - 1) {
		num_tile_items = num_entries - tile_offset;
		is_last_tile   = true;
	}

	extern __shared__ uint shared_buffer[];
	LoadDBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
	    col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

	__syncthreads();

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

	const char* filename = argv[1];
	const char* output_csv = argv[2];

	// Load data
	size_t n_tup = 0;
	uint32_t* h_org_arr_u32 = load_binary_file<uint32_t>(filename, n_tup);

	if (!h_org_arr_u32) {
		std::cerr << "Failed to load data file" << std::endl;
		return 1;
	}

	// Convert to int32
	int* h_org_arr = new int[n_tup];
	for (size_t i = 0; i < n_tup; i++) {
		h_org_arr[i] = (int)h_org_arr_u32[i];
	}
	delete[] h_org_arr_u32;

	// Ensure data size is multiple of 512 (128 * 4)
	size_t tile_size = 512;
	size_t aligned_n_tup = ((n_tup + tile_size - 1) / tile_size) * tile_size;
	if (aligned_n_tup != n_tup) {
		int* aligned_data = new int[aligned_n_tup];
		memcpy(aligned_data, h_org_arr, n_tup * sizeof(int));
		for (size_t i = n_tup; i < aligned_n_tup; i++) {
			aligned_data[i] = h_org_arr[n_tup - 1];
		}
		delete[] h_org_arr;
		h_org_arr = aligned_data;
		n_tup = aligned_n_tup;
	}

	int       block_size       = 128;
	int       elem_per_thread  = 4;
	int       num_blocks       = n_tup / block_size;
	auto*     encoded_data     = new int[n_tup * 2]();
	uint64_t  ofs_c            = num_blocks + 1;
	auto*     ofs_arr          = new uint[ofs_c]();
	auto*     copy_data        = new int[n_tup];

	/* Data needs to be copied. the encoding changes the original data. */
	memcpy(copy_data, h_org_arr, n_tup * sizeof(int));

	// Measure compression time
	auto encode_start = std::chrono::high_resolution_clock::now();
	uint encoded_data_bsz = deltaBinPack(copy_data, encoded_data, ofs_arr, n_tup);
	auto encode_end = std::chrono::high_resolution_clock::now();
	double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

	tile_based::encoded_column h_col {ofs_arr, reinterpret_cast<uint*>(encoded_data), n_tup * 4};
	uint* d_col_block_start = tile_based::loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1);
	uint* d_col_data        = tile_based::loadColumnToGPU<uint>(h_col.data, encoded_data_bsz);

	tile_based::encoded_column d_col {d_col_block_start, d_col_data};

	const int num_threads      = 128;
	const int items_per_thread = 4;
	int*      col              = nullptr;
	size_t    dg               = (n_tup + tile_size - 1) / tile_size;
	size_t    db               = num_threads;
	size_t    ns               = 3000;

	cudaMalloc((void**)&col, n_tup * sizeof(int));
	cudaDeviceSynchronize();

	// Warmup
	for (int i = 0; i < 3; i++) {
		runDBinKernel<num_threads, items_per_thread><<<dg, db, ns>>>(col, d_col.block_start, d_col.data, n_tup);
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
		runDBinKernel<num_threads, items_per_thread><<<dg, db, ns>>>(col, d_col.block_start, d_col.data, n_tup);
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
	int* temp = new int[n_tup];
	CubDebugExit(cudaMemcpy(temp, col, sizeof(int) * n_tup, cudaMemcpyDeviceToHost));

	bool correct = true;
	for (size_t i = 0; i < n_tup; i++) {
		if (h_org_arr[i] != temp[i]) {
			std::cout << "ERROR:" << i << " " << h_org_arr[i] << " " << temp[i] << '\n';
			correct = false;
			break;
		}
	}

	if (correct) {
		std::cout << "Verification passed!" << std::endl;
	}

	// Calculate metrics
	size_t original_size = n_tup * sizeof(int);
	size_t compressed_size = encoded_data_bsz * sizeof(uint);
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

	csv_file << "Delta,"
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
	delete[] h_org_arr;
	delete[] encoded_data;
	delete[] ofs_arr;
	delete[] copy_data;
	delete[] temp;
	cudaFree(col);
	cudaFree(d_col_block_start);
	cudaFree(d_col_data);

	return 0;
}
