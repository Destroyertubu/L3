#include "fastlanes.cuh"
#include "debug.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/rsum/rsum.cuh"
#include "fls_gen/transpose/transpose.hpp"
#include "fls_gen/unrsum/unrsum.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

// 正向转置表: TRANSPOSE[original_idx] = transposed_idx
__constant__ uint16_t TRANSPOSE[1024] = {
       0,  128,  256,  384,  512,  640,  768,  896,   64,  192,  320,  448,  576,  704,  832,  960,
      32,  160,  288,  416,  544,  672,  800,  928,   96,  224,  352,  480,  608,  736,  864,  992,
      16,  144,  272,  400,  528,  656,  784,  912,   80,  208,  336,  464,  592,  720,  848,  976,
      48,  176,  304,  432,  560,  688,  816,  944,  112,  240,  368,  496,  624,  752,  880, 1008,
       1,  129,  257,  385,  513,  641,  769,  897,   65,  193,  321,  449,  577,  705,  833,  961,
      33,  161,  289,  417,  545,  673,  801,  929,   97,  225,  353,  481,  609,  737,  865,  993,
      17,  145,  273,  401,  529,  657,  785,  913,   81,  209,  337,  465,  593,  721,  849,  977,
      49,  177,  305,  433,  561,  689,  817,  945,  113,  241,  369,  497,  625,  753,  881, 1009,
       2,  130,  258,  386,  514,  642,  770,  898,   66,  194,  322,  450,  578,  706,  834,  962,
      34,  162,  290,  418,  546,  674,  802,  930,   98,  226,  354,  482,  610,  738,  866,  994,
      18,  146,  274,  402,  530,  658,  786,  914,   82,  210,  338,  466,  594,  722,  850,  978,
      50,  178,  306,  434,  562,  690,  818,  946,  114,  242,  370,  498,  626,  754,  882, 1010,
       3,  131,  259,  387,  515,  643,  771,  899,   67,  195,  323,  451,  579,  707,  835,  963,
      35,  163,  291,  419,  547,  675,  803,  931,   99,  227,  355,  483,  611,  739,  867,  995,
      19,  147,  275,  403,  531,  659,  787,  915,   83,  211,  339,  467,  595,  723,  851,  979,
      51,  179,  307,  435,  563,  691,  819,  947,  115,  243,  371,  499,  627,  755,  883, 1011,
       4,  132,  260,  388,  516,  644,  772,  900,   68,  196,  324,  452,  580,  708,  836,  964,
      36,  164,  292,  420,  548,  676,  804,  932,  100,  228,  356,  484,  612,  740,  868,  996,
      20,  148,  276,  404,  532,  660,  788,  916,   84,  212,  340,  468,  596,  724,  852,  980,
      52,  180,  308,  436,  564,  692,  820,  948,  116,  244,  372,  500,  628,  756,  884, 1012,
       5,  133,  261,  389,  517,  645,  773,  901,   69,  197,  325,  453,  581,  709,  837,  965,
      37,  165,  293,  421,  549,  677,  805,  933,  101,  229,  357,  485,  613,  741,  869,  997,
      21,  149,  277,  405,  533,  661,  789,  917,   85,  213,  341,  469,  597,  725,  853,  981,
      53,  181,  309,  437,  565,  693,  821,  949,  117,  245,  373,  501,  629,  757,  885, 1013,
       6,  134,  262,  390,  518,  646,  774,  902,   70,  198,  326,  454,  582,  710,  838,  966,
      38,  166,  294,  422,  550,  678,  806,  934,  102,  230,  358,  486,  614,  742,  870,  998,
      22,  150,  278,  406,  534,  662,  790,  918,   86,  214,  342,  470,  598,  726,  854,  982,
      54,  182,  310,  438,  566,  694,  822,  950,  118,  246,  374,  502,  630,  758,  886, 1014,
       7,  135,  263,  391,  519,  647,  775,  903,   71,  199,  327,  455,  583,  711,  839,  967,
      39,  167,  295,  423,  551,  679,  807,  935,  103,  231,  359,  487,  615,  743,  871,  999,
      23,  151,  279,  407,  535,  663,  791,  919,   87,  215,  343,  471,  599,  727,  855,  983,
      55,  183,  311,  439,  567,  695,  823,  951,  119,  247,  375,  503,  631,  759,  887, 1015,
       8,  136,  264,  392,  520,  648,  776,  904,   72,  200,  328,  456,  584,  712,  840,  968,
      40,  168,  296,  424,  552,  680,  808,  936,  104,  232,  360,  488,  616,  744,  872, 1000,
      24,  152,  280,  408,  536,  664,  792,  920,   88,  216,  344,  472,  600,  728,  856,  984,
      56,  184,  312,  440,  568,  696,  824,  952,  120,  248,  376,  504,  632,  760,  888, 1016,
       9,  137,  265,  393,  521,  649,  777,  905,   73,  201,  329,  457,  585,  713,  841,  969,
      41,  169,  297,  425,  553,  681,  809,  937,  105,  233,  361,  489,  617,  745,  873, 1001,
      25,  153,  281,  409,  537,  665,  793,  921,   89,  217,  345,  473,  601,  729,  857,  985,
      57,  185,  313,  441,  569,  697,  825,  953,  121,  249,  377,  505,  633,  761,  889, 1017,
      10,  138,  266,  394,  522,  650,  778,  906,   74,  202,  330,  458,  586,  714,  842,  970,
      42,  170,  298,  426,  554,  682,  810,  938,  106,  234,  362,  490,  618,  746,  874, 1002,
      26,  154,  282,  410,  538,  666,  794,  922,   90,  218,  346,  474,  602,  730,  858,  986,
      58,  186,  314,  442,  570,  698,  826,  954,  122,  250,  378,  506,  634,  762,  890, 1018,
      11,  139,  267,  395,  523,  651,  779,  907,   75,  203,  331,  459,  587,  715,  843,  971,
      43,  171,  299,  427,  555,  683,  811,  939,  107,  235,  363,  491,  619,  747,  875, 1003,
      27,  155,  283,  411,  539,  667,  795,  923,   91,  219,  347,  475,  603,  731,  859,  987,
      59,  187,  315,  443,  571,  699,  827,  955,  123,  251,  379,  507,  635,  763,  891, 1019,
      12,  140,  268,  396,  524,  652,  780,  908,   76,  204,  332,  460,  588,  716,  844,  972,
      44,  172,  300,  428,  556,  684,  812,  940,  108,  236,  364,  492,  620,  748,  876, 1004,
      28,  156,  284,  412,  540,  668,  796,  924,   92,  220,  348,  476,  604,  732,  860,  988,
      60,  188,  316,  444,  572,  700,  828,  956,  124,  252,  380,  508,  636,  764,  892, 1020,
      13,  141,  269,  397,  525,  653,  781,  909,   77,  205,  333,  461,  589,  717,  845,  973,
      45,  173,  301,  429,  557,  685,  813,  941,  109,  237,  365,  493,  621,  749,  877, 1005,
      29,  157,  285,  413,  541,  669,  797,  925,   93,  221,  349,  477,  605,  733,  861,  989,
      61,  189,  317,  445,  573,  701,  829,  957,  125,  253,  381,  509,  637,  765,  893, 1021,
      14,  142,  270,  398,  526,  654,  782,  910,   78,  206,  334,  462,  590,  718,  846,  974,
      46,  174,  302,  430,  558,  686,  814,  942,  110,  238,  366,  494,  622,  750,  878, 1006,
      30,  158,  286,  414,  542,  670,  798,  926,   94,  222,  350,  478,  606,  734,  862,  990,
      62,  190,  318,  446,  574,  702,  830,  958,  126,  254,  382,  510,  638,  766,  894, 1022,
      15,  143,  271,  399,  527,  655,  783,  911,   79,  207,  335,  463,  591,  719,  847,  975,
      47,  175,  303,  431,  559,  687,  815,  943,  111,  239,  367,  495,  623,  751,  879, 1007,
      31,  159,  287,  415,  543,  671,  799,  927,   95,  223,  351,  479,  607,  735,  863,  991,
      63,  191,  319,  447,  575,  703,  831,  959,  127,  255,  383,  511,  639,  767,  895, 1023
};

// Delta解码内核（优化版）
__global__ void delta_decode_optimized(const uint32_t* __restrict in, uint32_t* __restrict out,
                                        const uint32_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + ((blc_idx * bw) << 5);
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum[1024];

	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_32(sm_unpacked, sm_rsum, base);
	__syncthreads();

	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = TRANSPOSE[original_idx];
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

uint8_t compute_max_delta_bitwidth(uint32_t* transposed_arr, uint32_t* unrsummed_arr,
                                    uint32_t* org_arr, uint64_t n_vec, uint64_t vec_sz) {
	uint32_t global_max_delta = 0;
	auto* in_ptr = org_arr;

	for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_ptr, transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(transposed_arr, unrsummed_arr);

		for (uint64_t i = 0; i < vec_sz; i++) {
			if (unrsummed_arr[i] > global_max_delta) {
				global_max_delta = unrsummed_arr[i];
			}
		}
		in_ptr += vec_sz;
	}

	if (global_max_delta == 0) return 1;
	return static_cast<uint8_t>(std::ceil(std::log2(global_max_delta + 1)));
}

struct BenchResult {
	std::string dataset;
	uint64_t elements;
	uint8_t bitwidth;
	double compression_ratio;
	double encode_time_ms;
	double decode_time_ms;
	double encode_gbps;
	double decode_gbps;
	bool verified;
};

BenchResult run_benchmark(const std::string& data_file, const std::string& dataset_name) {
	BenchResult result;
	result.dataset = dataset_name;
	result.verified = false;

	std::ifstream file(data_file, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << data_file << std::endl;
		return result;
	}
	uint64_t file_size = file.tellg();
	file.seekg(0, std::ios::beg);

	const uint64_t warp_sz = 32;
	const uint64_t vec_sz = 1024;
	uint64_t total_elements = file_size / sizeof(uint32_t);
	const uint64_t n_vec = total_elements / vec_sz;
	const uint64_t n_tup = vec_sz * n_vec;
	const uint64_t v_blc_sz = 1;
	const uint64_t n_blc = n_vec / v_blc_sz;
	const uint64_t n_trd = v_blc_sz * warp_sz;

	result.elements = n_tup;

	auto* h_org_arr = new uint32_t[n_tup];
	auto* h_encoded_data = new uint32_t[n_tup];
	auto* h_decoded_arr = new uint32_t[n_tup];
	auto* h_transposed_arr = new uint32_t[vec_sz];
	auto* h_unrsummed_arr = new uint32_t[vec_sz];
	auto* h_base_arr = new uint32_t[32 * n_vec];
	auto* h_expected_original = new uint32_t[n_tup];
	uint32_t* d_base_arr = nullptr;
	uint32_t* d_decoded_arr = nullptr;
	uint32_t* d_encoded_arr = nullptr;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint32_t) * n_tup));

	file.read(reinterpret_cast<char*>(h_org_arr), n_tup * sizeof(uint32_t));
	file.close();

	// Compute bitwidth
	uint8_t num_bits = compute_max_delta_bitwidth(h_transposed_arr, h_unrsummed_arr,
	                                               h_org_arr, n_vec, vec_sz);
	result.bitwidth = num_bits;

	// Encode
	cudaEvent_t enc_start, enc_stop;
	cudaEventCreate(&enc_start);
	cudaEventCreate(&enc_stop);

	cudaEventRecord(enc_start);
	auto in_als = h_org_arr;
	auto out_als = h_encoded_data;
	auto base_als = h_base_arr;
	auto exp_orig = h_expected_original;

	for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_als, h_transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(h_transposed_arr, h_unrsummed_arr);
		std::memcpy(base_als, h_transposed_arr, sizeof(uint32_t) * 32);
		std::memcpy(exp_orig, in_als, sizeof(uint32_t) * vec_sz);
		generated::pack::fallback::scalar::pack(h_unrsummed_arr, out_als, num_bits);

		in_als = in_als + vec_sz;
		out_als = out_als + (num_bits * vec_sz / 32);
		base_als = base_als + 32;
		exp_orig = exp_orig + vec_sz;
	}
	cudaEventRecord(enc_stop);
	cudaEventSynchronize(enc_stop);

	float encode_time = 0;
	cudaEventElapsedTime(&encode_time, enc_start, enc_stop);
	result.encode_time_ms = encode_time;

	uint64_t original_size = n_tup * sizeof(uint32_t);
	uint64_t encoded_size = n_vec * (num_bits * vec_sz / 32) * sizeof(uint32_t);
	uint64_t base_size = 32 * n_vec * sizeof(uint32_t);
	uint64_t total_compressed_size = encoded_size + base_size;
	result.compression_ratio = (double)original_size / total_compressed_size;
	result.encode_gbps = (original_size / 1e9) / (encode_time / 1000.0);

	// Load to GPU
	d_encoded_arr = fastlanes::gpu::load_arr(h_encoded_data, encoded_size);
	d_base_arr = fastlanes::gpu::load_arr(h_base_arr, 32 * n_vec * sizeof(uint32_t));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// Decode benchmark
	const int num_runs = 10;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warmup
	delta_decode_optimized<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_optimized<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time = 0;
	cudaEventElapsedTime(&decode_time, start, stop);
	decode_time /= num_runs;
	result.decode_time_ms = decode_time;
	result.decode_gbps = (original_size / 1e9) / (decode_time / 1000.0);

	// Verify
	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	uint64_t errors = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) {
			errors++;
		}
	}
	result.verified = (errors == 0);

	// Cleanup
	cudaEventDestroy(enc_start);
	cudaEventDestroy(enc_stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	delete[] h_org_arr;
	delete[] h_encoded_data;
	delete[] h_decoded_arr;
	delete[] h_transposed_arr;
	delete[] h_unrsummed_arr;
	delete[] h_base_arr;
	delete[] h_expected_original;

	CUDA_SAFE_CALL(cudaFree(d_decoded_arr));
	CUDA_SAFE_CALL(cudaFree(d_encoded_arr));
	CUDA_SAFE_CALL(cudaFree(d_base_arr));

	return result;
}

int main(int argc, char** argv) {
	std::string data_dir = "/home/xiayouyang/code/L3/data/sosd";
	std::string output_file = "";

	// Parse arguments
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--data-dir" && i + 1 < argc) {
			data_dir = argv[++i];
		} else if (arg == "--output" && i + 1 < argc) {
			output_file = argv[++i];
		}
	}

	// uint32 datasets only (FastLanes GPU only supports uint32)
	std::vector<std::pair<std::string, std::string>> datasets = {
		{"1-linear", "1-linear_200M_uint32.bin"},
		{"2-normal", "2-normal_200M_uint32.bin"},
		{"5-books", "5-books_200M_uint32.bin"},
		{"9-movieid", "9-movieid_uint32.bin"},
		{"14-cosmos", "14-cosmos_int32.bin"},
		{"18-site", "18-site_250k_uint32.bin"},
		{"19-weight", "19-weight_25k_uint32.bin"},
		{"20-adult", "20-adult_30k_uint32.bin"},
	};

	cudaDeviceSynchronize();
	std::cout << "FastLanes GPU Benchmark - Multi-Dataset\n";
	std::cout << "========================================\n";
	std::cout << "Data directory: " << data_dir << "\n\n";

	std::vector<BenchResult> results;

	for (const auto& [name, filename] : datasets) {
		std::string filepath = data_dir + "/" + filename;
		std::cout << "Testing: " << name << " (" << filename << ")...\n";

		BenchResult result = run_benchmark(filepath, name);
		results.push_back(result);

		std::cout << "  Elements: " << result.elements
		          << ", Bitwidth: " << (int)result.bitwidth
		          << ", Ratio: " << result.compression_ratio
		          << "x, Decode: " << result.decode_gbps << " GB/s"
		          << ", Verify: " << (result.verified ? "OK" : "FAIL")
		          << "\n";
	}

	// Print summary
	std::cout << "\n========================================\n";
	std::cout << "Summary:\n";
	std::cout << "========================================\n";
	printf("%-15s %12s %8s %10s %12s %12s %8s\n",
	       "Dataset", "Elements", "Bits", "Ratio", "Enc GB/s", "Dec GB/s", "Verify");
	printf("-------------------------------------------------------------------------------\n");

	for (const auto& r : results) {
		printf("%-15s %12lu %8d %10.2f %12.2f %12.2f %8s\n",
		       r.dataset.c_str(), r.elements, r.bitwidth,
		       r.compression_ratio, r.encode_gbps, r.decode_gbps,
		       r.verified ? "OK" : "FAIL");
	}

	// Write CSV if output specified
	if (!output_file.empty()) {
		std::ofstream csv(output_file);
		csv << "Dataset,Elements,Bitwidth,CompressionRatio,EncodeGBps,DecodeGBps,Verified\n";
		for (const auto& r : results) {
			csv << r.dataset << ","
			    << r.elements << ","
			    << (int)r.bitwidth << ","
			    << r.compression_ratio << ","
			    << r.encode_gbps << ","
			    << r.decode_gbps << ","
			    << (r.verified ? "OK" : "FAIL") << "\n";
		}
		csv.close();
		std::cout << "\nResults written to: " << output_file << "\n";
	}

	return 0;
}
