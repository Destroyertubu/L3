#include "fastlanes.cuh"
#include "debug.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/rsum/rsum.cuh"
#include "fls_gen/rsum/rsum_64.cuh"
#include "fls_gen/transpose/transpose.hpp"
#include "fls_gen/unrsum/unrsum.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "fls_gen/unpack/unpack_64.cuh"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

// 正向转置表: TRANSPOSE[original_idx] = transposed_idx
// 用于优化版本：随机读共享内存，顺序写全局内存
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

// 逆转置映射表: INVERSE_TRANSPOSE[transposed_idx] = original_idx
__constant__ uint16_t INVERSE_TRANSPOSE[1024] = {
       0,   64,  128,  192,  256,  320,  384,  448,  512,  576,  640,  704,  768,  832,  896,  960,
      32,   96,  160,  224,  288,  352,  416,  480,  544,  608,  672,  736,  800,  864,  928,  992,
      16,   80,  144,  208,  272,  336,  400,  464,  528,  592,  656,  720,  784,  848,  912,  976,
      48,  112,  176,  240,  304,  368,  432,  496,  560,  624,  688,  752,  816,  880,  944, 1008,
       8,   72,  136,  200,  264,  328,  392,  456,  520,  584,  648,  712,  776,  840,  904,  968,
      40,  104,  168,  232,  296,  360,  424,  488,  552,  616,  680,  744,  808,  872,  936, 1000,
      24,   88,  152,  216,  280,  344,  408,  472,  536,  600,  664,  728,  792,  856,  920,  984,
      56,  120,  184,  248,  312,  376,  440,  504,  568,  632,  696,  760,  824,  888,  952, 1016,
       1,   65,  129,  193,  257,  321,  385,  449,  513,  577,  641,  705,  769,  833,  897,  961,
      33,   97,  161,  225,  289,  353,  417,  481,  545,  609,  673,  737,  801,  865,  929,  993,
      17,   81,  145,  209,  273,  337,  401,  465,  529,  593,  657,  721,  785,  849,  913,  977,
      49,  113,  177,  241,  305,  369,  433,  497,  561,  625,  689,  753,  817,  881,  945, 1009,
       9,   73,  137,  201,  265,  329,  393,  457,  521,  585,  649,  713,  777,  841,  905,  969,
      41,  105,  169,  233,  297,  361,  425,  489,  553,  617,  681,  745,  809,  873,  937, 1001,
      25,   89,  153,  217,  281,  345,  409,  473,  537,  601,  665,  729,  793,  857,  921,  985,
      57,  121,  185,  249,  313,  377,  441,  505,  569,  633,  697,  761,  825,  889,  953, 1017,
       2,   66,  130,  194,  258,  322,  386,  450,  514,  578,  642,  706,  770,  834,  898,  962,
      34,   98,  162,  226,  290,  354,  418,  482,  546,  610,  674,  738,  802,  866,  930,  994,
      18,   82,  146,  210,  274,  338,  402,  466,  530,  594,  658,  722,  786,  850,  914,  978,
      50,  114,  178,  242,  306,  370,  434,  498,  562,  626,  690,  754,  818,  882,  946, 1010,
      10,   74,  138,  202,  266,  330,  394,  458,  522,  586,  650,  714,  778,  842,  906,  970,
      42,  106,  170,  234,  298,  362,  426,  490,  554,  618,  682,  746,  810,  874,  938, 1002,
      26,   90,  154,  218,  282,  346,  410,  474,  538,  602,  666,  730,  794,  858,  922,  986,
      58,  122,  186,  250,  314,  378,  442,  506,  570,  634,  698,  762,  826,  890,  954, 1018,
       3,   67,  131,  195,  259,  323,  387,  451,  515,  579,  643,  707,  771,  835,  899,  963,
      35,   99,  163,  227,  291,  355,  419,  483,  547,  611,  675,  739,  803,  867,  931,  995,
      19,   83,  147,  211,  275,  339,  403,  467,  531,  595,  659,  723,  787,  851,  915,  979,
      51,  115,  179,  243,  307,  371,  435,  499,  563,  627,  691,  755,  819,  883,  947, 1011,
      11,   75,  139,  203,  267,  331,  395,  459,  523,  587,  651,  715,  779,  843,  907,  971,
      43,  107,  171,  235,  299,  363,  427,  491,  555,  619,  683,  747,  811,  875,  939, 1003,
      27,   91,  155,  219,  283,  347,  411,  475,  539,  603,  667,  731,  795,  859,  923,  987,
      59,  123,  187,  251,  315,  379,  443,  507,  571,  635,  699,  763,  827,  891,  955, 1019,
       4,   68,  132,  196,  260,  324,  388,  452,  516,  580,  644,  708,  772,  836,  900,  964,
      36,  100,  164,  228,  292,  356,  420,  484,  548,  612,  676,  740,  804,  868,  932,  996,
      20,   84,  148,  212,  276,  340,  404,  468,  532,  596,  660,  724,  788,  852,  916,  980,
      52,  116,  180,  244,  308,  372,  436,  500,  564,  628,  692,  756,  820,  884,  948, 1012,
      12,   76,  140,  204,  268,  332,  396,  460,  524,  588,  652,  716,  780,  844,  908,  972,
      44,  108,  172,  236,  300,  364,  428,  492,  556,  620,  684,  748,  812,  876,  940, 1004,
      28,   92,  156,  220,  284,  348,  412,  476,  540,  604,  668,  732,  796,  860,  924,  988,
      60,  124,  188,  252,  316,  380,  444,  508,  572,  636,  700,  764,  828,  892,  956, 1020,
       5,   69,  133,  197,  261,  325,  389,  453,  517,  581,  645,  709,  773,  837,  901,  965,
      37,  101,  165,  229,  293,  357,  421,  485,  549,  613,  677,  741,  805,  869,  933,  997,
      21,   85,  149,  213,  277,  341,  405,  469,  533,  597,  661,  725,  789,  853,  917,  981,
      53,  117,  181,  245,  309,  373,  437,  501,  565,  629,  693,  757,  821,  885,  949, 1013,
      13,   77,  141,  205,  269,  333,  397,  461,  525,  589,  653,  717,  781,  845,  909,  973,
      45,  109,  173,  237,  301,  365,  429,  493,  557,  621,  685,  749,  813,  877,  941, 1005,
      29,   93,  157,  221,  285,  349,  413,  477,  541,  605,  669,  733,  797,  861,  925,  989,
      61,  125,  189,  253,  317,  381,  445,  509,  573,  637,  701,  765,  829,  893,  957, 1021,
       6,   70,  134,  198,  262,  326,  390,  454,  518,  582,  646,  710,  774,  838,  902,  966,
      38,  102,  166,  230,  294,  358,  422,  486,  550,  614,  678,  742,  806,  870,  934,  998,
      22,   86,  150,  214,  278,  342,  406,  470,  534,  598,  662,  726,  790,  854,  918,  982,
      54,  118,  182,  246,  310,  374,  438,  502,  566,  630,  694,  758,  822,  886,  950, 1014,
      14,   78,  142,  206,  270,  334,  398,  462,  526,  590,  654,  718,  782,  846,  910,  974,
      46,  110,  174,  238,  302,  366,  430,  494,  558,  622,  686,  750,  814,  878,  942, 1006,
      30,   94,  158,  222,  286,  350,  414,  478,  542,  606,  670,  734,  798,  862,  926,  990,
      62,  126,  190,  254,  318,  382,  446,  510,  574,  638,  702,  766,  830,  894,  958, 1022,
       7,   71,  135,  199,  263,  327,  391,  455,  519,  583,  647,  711,  775,  839,  903,  967,
      39,  103,  167,  231,  295,  359,  423,  487,  551,  615,  679,  743,  807,  871,  935,  999,
      23,   87,  151,  215,  279,  343,  407,  471,  535,  599,  663,  727,  791,  855,  919,  983,
      55,  119,  183,  247,  311,  375,  439,  503,  567,  631,  695,  759,  823,  887,  951, 1015,
      15,   79,  143,  207,  271,  335,  399,  463,  527,  591,  655,  719,  783,  847,  911,  975,
      47,  111,  175,  239,  303,  367,  431,  495,  559,  623,  687,  751,  815,  879,  943, 1007,
      31,   95,  159,  223,  287,  351,  415,  479,  543,  607,  671,  735,  799,  863,  927,  991,
      63,  127,  191,  255,  319,  383,  447,  511,  575,  639,  703,  767,  831,  895,  959, 1023
};

// ===== Per-Block Bitwidth Kernels =====

// Delta解码内核（per-block位宽，无逆转置）
__global__ void delta_decode_per_block_no_transpose(const uint32_t* __restrict in, uint32_t* __restrict out,
                                                      const uint32_t* __restrict base,
                                                      const uint8_t* __restrict bitwidths,
                                                      const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_arr[1024];

	// Step 1: Unpack to shared memory
	unpack_device(in, sm_arr, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) directly to output
	d_rsum_32(sm_arr, out, base);
}

// ===== 辅助函数（用于计算公式版本）=====
// 3-bit 反转查找表
__device__ __forceinline__ uint32_t bit_reverse_3_helper(uint32_t x) {
	constexpr uint32_t lut = 0x73516240;
	return (lut >> (x * 4)) & 0x7;
}

// 计算正向转置索引（不使用查找表）
__device__ __forceinline__ uint32_t compute_transpose_idx(uint32_t o) {
	uint32_t low3 = o & 7;
	uint32_t mid3 = (o >> 3) & 7;
	uint32_t high4 = o >> 6;
	uint32_t row = (low3 << 3) | bit_reverse_3_helper(mid3);
	return (row << 4) | high4;
}

// Delta解码内核（per-block位宽，含逆转置，优化版）
__global__ void delta_decode_per_block_optimized(const uint32_t* __restrict in, uint32_t* __restrict out,
                                                   const uint32_t* __restrict base,
                                                   const uint8_t* __restrict bitwidths,
                                                   const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum[1024];

	// Step 1: Unpack to shared memory
	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) to second shared memory
	d_rsum_32(sm_unpacked, sm_rsum, base);
	__syncthreads();

	// Step 3: 随机读共享内存，顺序写全局内存（coalesced）
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = TRANSPOSE[original_idx];
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// ===== Per-Block COMPUTE ONLY 版本 (无padding，只用计算公式) =====
__global__ void delta_decode_per_block_compute_only(const uint32_t* __restrict in, uint32_t* __restrict out,
                                                     const uint32_t* __restrict base,
                                                     const uint8_t* __restrict bitwidths,
                                                     const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum[1024];

	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_32(sm_unpacked, sm_rsum, base);
	__syncthreads();

	// 只用计算公式，无padding
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// ===== Per-Block COMPUTE + PADDED 版本 (最优化) =====
__global__ void delta_decode_per_block_compute_padded(const uint32_t* __restrict in, uint32_t* __restrict out,
                                                       const uint32_t* __restrict base,
                                                       const uint8_t* __restrict bitwidths,
                                                       const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum_padded[1024 + 32];

	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_32(sm_unpacked, sm_unpacked, base);
	__syncthreads();

	// 复制到带padding的数组
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t idx = trd_idx + i * 32;
		uint32_t padded_idx = idx + (idx / 32);
		sm_rsum_padded[padded_idx] = sm_unpacked[idx];
	}
	__syncthreads();

	// 计算公式 + padding
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		uint32_t padded_transposed = transposed_idx + (transposed_idx / 32);
		out[original_idx] = sm_rsum_padded[padded_transposed];
	}
}

// ===== Uniform Bitwidth Kernels (Original) =====

// Delta解码内核（无逆转置）：unpack + rsum，输出转置顺序
__global__ void delta_decode_no_transpose(const uint32_t* __restrict in, uint32_t* __restrict out,
                                           const uint32_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	in   = in + ((blc_idx * bw) << 5);
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_arr[1024];

	// Step 1: Unpack to shared memory
	unpack_device(in, sm_arr, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) directly to output
	d_rsum_32(sm_arr, out, base);
}

// Delta解码内核（含逆转置）：unpack + rsum + inverse_transpose，输出原始顺序
// 原始版本：随机写全局内存（非coalesced）
__global__ void delta_decode_with_transpose(const uint32_t* __restrict in, uint32_t* __restrict out,
                                             const uint32_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + ((blc_idx * bw) << 5);
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum[1024];

	// Step 1: Unpack to shared memory
	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) to second shared memory
	d_rsum_32(sm_unpacked, sm_rsum, base);
	__syncthreads();

	// Step 3: Inverse transpose from sm_rsum to output (随机写)
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t transposed_idx = trd_idx + i * 32;
		uint32_t original_idx = INVERSE_TRANSPOSE[transposed_idx];
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// Delta解码内核（优化版逆转置）：unpack + rsum + transpose
// 优化策略：随机读共享内存 + 顺序写全局内存（coalesced）
__global__ void delta_decode_optimized(const uint32_t* __restrict in, uint32_t* __restrict out,
                                        const uint32_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + ((blc_idx * bw) << 5);
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum[1024];

	// Step 1: Unpack to shared memory
	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) to second shared memory
	d_rsum_32(sm_unpacked, sm_rsum, base);
	__syncthreads();

	// Step 3: 随机读共享内存，顺序写全局内存（coalesced）
	// 使用正向转置表：给定 original_idx，找到对应的 transposed_idx
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;  // 顺序写入位置
		uint32_t transposed_idx = TRANSPOSE[original_idx];  // 查找源位置
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// 3-bit 反转查找表（用于计算公式版本）
__device__ __forceinline__ uint32_t bit_reverse_3(uint32_t x) {
	// 0->0, 1->4, 2->2, 3->6, 4->1, 5->5, 6->3, 7->7
	constexpr uint32_t lut = 0x73516240;  // packed lookup table
	return (lut >> (x * 4)) & 0x7;
}

// 优化版本2：使用计算公式代替查表
__global__ void delta_decode_compute(const uint32_t* __restrict in, uint32_t* __restrict out,
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

	// 使用计算公式代替查表
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// 优化版本3：共享内存 padding 避免 bank conflicts
// 共享内存有32个bank，每个bank 4字节
// 添加 padding 使得连续访问跨越不同 bank
__global__ void delta_decode_padded(const uint32_t* __restrict in, uint32_t* __restrict out,
                                     const uint32_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + ((blc_idx * bw) << 5);
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	// 每32个元素后添加1个padding，避免bank冲突
	// 1024个元素 + 32个padding = 1056
	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum_padded[1024 + 32];

	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	// 复制到带padding的数组（d_rsum_32输出到临时位置）
	d_rsum_32(sm_unpacked, sm_unpacked, base);  // 原地操作
	__syncthreads();

	// 复制到带padding的数组
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t idx = trd_idx + i * 32;
		uint32_t padded_idx = idx + (idx / 32);  // 每32个元素后加1
		sm_rsum_padded[padded_idx] = sm_unpacked[idx];
	}
	__syncthreads();

	// 随机读带padding的共享内存
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = TRANSPOSE[original_idx];
		uint32_t padded_transposed = transposed_idx + (transposed_idx / 32);
		out[original_idx] = sm_rsum_padded[padded_transposed];
	}
}

// 优化版本4：计算公式 + padding 组合
__global__ void delta_decode_compute_padded(const uint32_t* __restrict in, uint32_t* __restrict out,
                                             const uint32_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + ((blc_idx * bw) << 5);
	out  = out + (blc_idx << 10);
	base = base + (blc_idx * 32);

	__shared__ uint32_t sm_unpacked[1024];
	__shared__ uint32_t sm_rsum_padded[1024 + 32];

	unpack_device(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_32(sm_unpacked, sm_unpacked, base);
	__syncthreads();

	// 复制到带padding的数组
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t idx = trd_idx + i * 32;
		uint32_t padded_idx = idx + (idx / 32);
		sm_rsum_padded[padded_idx] = sm_unpacked[idx];
	}
	__syncthreads();

	// 计算公式 + padding
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		uint32_t padded_transposed = transposed_idx + (transposed_idx / 32);
		out[original_idx] = sm_rsum_padded[padded_transposed];
	}
}

// ============================================================================
// 优化版本5：融合rsum+逆转置，直接写全局内存
// 核心思想：rsum计算时直接写到正确的输出位置，省去中间共享内存
// ============================================================================
__device__ __forceinline__ void d_rsum_32_fused_output(
    const uint32_t* __restrict__ sm_in,
    uint32_t* __restrict__ global_out,
    const uint32_t* __restrict__ base) {

    uint32_t trd_idx = threadIdx.x % 32;
    uint32_t r_0, r_1;

    // rsum处理顺序: 0,128,256,384,512,640,768,896, 64,192,320,448,576,704,832,960, 32,160,...
    // 每次计算后直接写到INVERSE_TRANSPOSE位置

    #define FUSED_RSUM_STEP(offset) \
        r_0 = sm_in[trd_idx + offset]; \
        r_1 = r_1 + r_0; \
        global_out[INVERSE_TRANSPOSE[trd_idx + offset]] = r_1;

    r_0 = sm_in[trd_idx + 0];
    r_1 = base[trd_idx] + r_0;
    global_out[INVERSE_TRANSPOSE[trd_idx + 0]] = r_1;

    FUSED_RSUM_STEP(128)
    FUSED_RSUM_STEP(256)
    FUSED_RSUM_STEP(384)
    FUSED_RSUM_STEP(512)
    FUSED_RSUM_STEP(640)
    FUSED_RSUM_STEP(768)
    FUSED_RSUM_STEP(896)
    FUSED_RSUM_STEP(64)
    FUSED_RSUM_STEP(192)
    FUSED_RSUM_STEP(320)
    FUSED_RSUM_STEP(448)
    FUSED_RSUM_STEP(576)
    FUSED_RSUM_STEP(704)
    FUSED_RSUM_STEP(832)
    FUSED_RSUM_STEP(960)
    FUSED_RSUM_STEP(32)
    FUSED_RSUM_STEP(160)
    FUSED_RSUM_STEP(288)
    FUSED_RSUM_STEP(416)
    FUSED_RSUM_STEP(544)
    FUSED_RSUM_STEP(672)
    FUSED_RSUM_STEP(800)
    FUSED_RSUM_STEP(928)
    FUSED_RSUM_STEP(96)
    FUSED_RSUM_STEP(224)
    FUSED_RSUM_STEP(352)
    FUSED_RSUM_STEP(480)
    FUSED_RSUM_STEP(608)
    FUSED_RSUM_STEP(736)
    FUSED_RSUM_STEP(864)
    FUSED_RSUM_STEP(992)

    #undef FUSED_RSUM_STEP
}

__global__ void delta_decode_fused(const uint32_t* __restrict in, uint32_t* __restrict out,
                                    const uint32_t* __restrict base, uint8_t bw) {
    uint32_t blc_idx = blockIdx.x;
    uint32_t trd_idx = threadIdx.x;
    in   = in + ((blc_idx * bw) << 5);
    out  = out + (blc_idx << 10);
    base = base + (blc_idx * 32);

    __shared__ uint32_t sm_unpacked[1024];

    // Step 1: Unpack到共享内存
    unpack_device(in, sm_unpacked, bw);
    __syncthreads();

    // Step 2: 融合rsum+逆转置，直接写全局内存
    d_rsum_32_fused_output(sm_unpacked, out, base);
}

// ============================================================================
// 优化版本6：寄存器暂存 + 批量写入（尝试提高写合并）
// ============================================================================
__global__ void delta_decode_reg_buffer(const uint32_t* __restrict in, uint32_t* __restrict out,
                                         const uint32_t* __restrict base, uint8_t bw) {
    uint32_t blc_idx = blockIdx.x;
    uint32_t trd_idx = threadIdx.x;
    in   = in + ((blc_idx * bw) << 5);
    out  = out + (blc_idx << 10);
    base = base + (blc_idx * 32);

    __shared__ uint32_t sm_unpacked[1024];

    unpack_device(in, sm_unpacked, bw);
    __syncthreads();

    // 在寄存器中计算所有32个rsum值
    uint32_t results[32];
    uint32_t r1 = base[trd_idx];

    // rsum计算顺序
    static const uint16_t RSUM_ORDER[32] = {
        0, 128, 256, 384, 512, 640, 768, 896,
        64, 192, 320, 448, 576, 704, 832, 960,
        32, 160, 288, 416, 544, 672, 800, 928,
        96, 224, 352, 480, 608, 736, 864, 992
    };

    #pragma unroll
    for (int k = 0; k < 32; k++) {
        r1 += sm_unpacked[trd_idx + RSUM_ORDER[k]];
        results[k] = r1;
    }

    // 批量写入：每个线程写32个值到正确位置
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        uint32_t transposed_pos = trd_idx + RSUM_ORDER[k];
        out[INVERSE_TRANSPOSE[transposed_pos]] = results[k];
    }
}

// ============================================================================
// 优化版本7：Warp协作写入（尝试改善写合并）
// 每轮32个线程协作写连续的32个位置
// ============================================================================
__global__ void delta_decode_warp_write(const uint32_t* __restrict in, uint32_t* __restrict out,
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

    // Warp协作：每轮写32个连续位置
    // 位置i需要的数据来自sm_rsum[TRANSPOSE[i]]
    #pragma unroll
    for (int round = 0; round < 32; round++) {
        uint32_t global_pos = round * 32 + trd_idx;  // 连续位置
        uint32_t src_pos = TRANSPOSE[global_pos];
        out[global_pos] = sm_rsum[src_pos];
    }
}

// 计算数据中所有delta值的最大位宽（全局统一位宽 - 保留用于对比）
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

// 计算每个block的独立位宽
void compute_per_block_bitwidths(uint32_t* transposed_arr, uint32_t* unrsummed_arr,
                                  uint32_t* org_arr, uint64_t n_vec, uint64_t vec_sz,
                                  uint8_t* bitwidths, uint64_t* offsets) {
	auto* in_ptr = org_arr;
	uint64_t current_offset = 0;

	for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_ptr, transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(transposed_arr, unrsummed_arr);

		// 找这个block的最大delta
		uint32_t block_max_delta = 0;
		for (uint64_t i = 0; i < vec_sz; i++) {
			if (unrsummed_arr[i] > block_max_delta) {
				block_max_delta = unrsummed_arr[i];
			}
		}

		// 计算这个block需要的位宽
		uint8_t bw = (block_max_delta == 0) ? 1 : static_cast<uint8_t>(std::ceil(std::log2(block_max_delta + 1)));
		bitwidths[vec_idx] = bw;

		// 记录这个block的数据偏移（以uint32为单位）
		offsets[vec_idx] = current_offset;
		current_offset += (bw * vec_sz / 32);  // 每个block的压缩数据大小

		in_ptr += vec_sz;
	}
	// 最后一个偏移用于计算总大小
	offsets[n_vec] = current_offset;
}

// ===== uint64 Delta Support =====

// Delta解码内核 uint64（无逆转置）：unpack + rsum，输出转置顺序
__global__ void delta_decode_no_transpose_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                              const uint64_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	// uint64: 16 lanes per block for pack, but 128 lanes for rsum
	in   = in + (blc_idx * bw * 16);
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);  // 16 chains for uint64 rsum

	__shared__ uint64_t sm_arr[1024];

	// Step 1: Unpack to shared memory
	unpack_device_64(in, sm_arr, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) directly to output
	d_rsum_64(sm_arr, out, base);
}

// Delta解码内核 uint64（优化版逆转置）：unpack + rsum + transpose
// 优化策略：随机读共享内存 + 顺序写全局内存（coalesced）
__global__ void delta_decode_optimized_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                           const uint64_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + (blc_idx * bw * 16);
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);

	__shared__ uint64_t sm_unpacked[1024];
	__shared__ uint64_t sm_rsum[1024];

	// Step 1: Unpack to shared memory
	unpack_device_64(in, sm_unpacked, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) to second shared memory
	d_rsum_64(sm_unpacked, sm_rsum, base);
	__syncthreads();

	// Step 3: 随机读共享内存，顺序写全局内存（coalesced）
	// 使用正向转置表：给定 original_idx，找到对应的 transposed_idx
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;  // 顺序写入位置
		uint32_t transposed_idx = TRANSPOSE[original_idx];  // 查找源位置
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// ============================================================================
// Delta解码内核 uint64 COMPUTE ONLY 版本（无padding，只用计算公式）
// ============================================================================
__global__ void delta_decode_compute_only_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                              const uint64_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + (blc_idx * bw * 16);
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);

	__shared__ uint64_t sm_unpacked[1024];
	__shared__ uint64_t sm_rsum[1024];

	unpack_device_64(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_64(sm_unpacked, sm_rsum, base);
	__syncthreads();

	// 只用计算公式，无padding
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// ============================================================================
// Delta解码内核 uint64 COMPUTE+PADDED 版本（最优化）
// 使用计算公式 + shared memory padding 消除bank冲突
// ============================================================================
__global__ void delta_decode_compute_padded_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                                const uint64_t* __restrict base, uint8_t bw) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	in   = in + (blc_idx * bw * 16);
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);

	__shared__ uint64_t sm_unpacked[1024];
	// 与uint32相同策略：每32个元素后加1个padding
	__shared__ uint64_t sm_rsum_padded[1024 + 32];

	// Step 1: Unpack to shared memory
	unpack_device_64(in, sm_unpacked, bw);
	__syncthreads();

	// Step 2: Prefix sum (rsum) in-place
	d_rsum_64(sm_unpacked, sm_unpacked, base);
	__syncthreads();

	// Step 3: 复制到带padding的数组 (每32个元素后加1个padding)
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t idx = trd_idx + i * 32;
		uint32_t padded_idx = idx + (idx / 32);
		sm_rsum_padded[padded_idx] = sm_unpacked[idx];
	}
	__syncthreads();

	// Step 4: 计算公式 + padding读取
	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		uint32_t padded_transposed = transposed_idx + (transposed_idx / 32);
		out[original_idx] = sm_rsum_padded[padded_transposed];
	}
}

// ============================================================================
// uint64 Per-Block Bitwidth Kernels
// ============================================================================

// Delta解码内核 uint64 per-block（无逆转置）
__global__ void delta_decode_per_block_no_transpose_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                                        const uint64_t* __restrict base,
                                                        const uint8_t* __restrict bitwidths,
                                                        const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);

	__shared__ uint64_t sm_arr[1024];

	unpack_device_64(in, sm_arr, bw);
	__syncthreads();

	d_rsum_64(sm_arr, out, base);
}

// Delta解码内核 uint64 per-block（含逆转置，优化版）
__global__ void delta_decode_per_block_optimized_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                                     const uint64_t* __restrict base,
                                                     const uint8_t* __restrict bitwidths,
                                                     const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);

	__shared__ uint64_t sm_unpacked[1024];
	__shared__ uint64_t sm_rsum[1024];

	unpack_device_64(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_64(sm_unpacked, sm_rsum, base);
	__syncthreads();

	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = TRANSPOSE[original_idx];
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// Delta解码内核 uint64 per-block COMPUTE ONLY（无padding）
__global__ void delta_decode_per_block_compute_only_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                                        const uint64_t* __restrict base,
                                                        const uint8_t* __restrict bitwidths,
                                                        const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);

	__shared__ uint64_t sm_unpacked[1024];
	__shared__ uint64_t sm_rsum[1024];

	unpack_device_64(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_64(sm_unpacked, sm_rsum, base);
	__syncthreads();

	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		out[original_idx] = sm_rsum[transposed_idx];
	}
}

// Delta解码内核 uint64 per-block COMPUTE+PADDED（最优化）
__global__ void delta_decode_per_block_compute_padded_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                                          const uint64_t* __restrict base,
                                                          const uint8_t* __restrict bitwidths,
                                                          const uint64_t* __restrict offsets) {
	uint32_t blc_idx = blockIdx.x;
	uint32_t trd_idx = threadIdx.x;
	uint8_t bw = bitwidths[blc_idx];
	in   = in + offsets[blc_idx];
	out  = out + (blc_idx * 1024);
	base = base + (blc_idx * 16);

	__shared__ uint64_t sm_unpacked[1024];
	__shared__ uint64_t sm_rsum_padded[1024 + 32];

	unpack_device_64(in, sm_unpacked, bw);
	__syncthreads();

	d_rsum_64(sm_unpacked, sm_unpacked, base);
	__syncthreads();

	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t idx = trd_idx + i * 32;
		uint32_t padded_idx = idx + (idx / 32);
		sm_rsum_padded[padded_idx] = sm_unpacked[idx];
	}
	__syncthreads();

	#pragma unroll
	for (int i = 0; i < 32; i++) {
		uint32_t original_idx = trd_idx + i * 32;
		uint32_t transposed_idx = compute_transpose_idx(original_idx);
		uint32_t padded_transposed = transposed_idx + (transposed_idx / 32);
		out[original_idx] = sm_rsum_padded[padded_transposed];
	}
}

// 计算uint64数据中所有delta值的最大位宽
uint8_t compute_max_delta_bitwidth_64(uint64_t* transposed_arr, uint64_t* unrsummed_arr,
                                       uint64_t* org_arr, uint64_t n_vec, uint64_t vec_sz) {
	uint64_t global_max_delta = 0;
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
	return static_cast<uint8_t>(std::ceil(std::log2(static_cast<double>(global_max_delta) + 1)));
}

// 计算uint64每个block的独立位宽
void compute_per_block_bitwidths_64(uint64_t* transposed_arr, uint64_t* unrsummed_arr,
                                     uint64_t* org_arr, uint64_t n_vec, uint64_t vec_sz,
                                     uint8_t* bitwidths, uint64_t* offsets) {
	auto* in_ptr = org_arr;
	uint64_t current_offset = 0;

	for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_ptr, transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(transposed_arr, unrsummed_arr);

		uint64_t block_max_delta = 0;
		for (uint64_t i = 0; i < vec_sz; i++) {
			if (unrsummed_arr[i] > block_max_delta) {
				block_max_delta = unrsummed_arr[i];
			}
		}

		uint8_t bw = (block_max_delta == 0) ? 1 : static_cast<uint8_t>(std::ceil(std::log2(static_cast<double>(block_max_delta) + 1)));
		bitwidths[vec_idx] = bw;

		offsets[vec_idx] = current_offset;
		current_offset += (bw * vec_sz / 64);  // uint64: 除以64

		in_ptr += vec_sz;
	}
	offsets[n_vec] = current_offset;
}

int run_benchmark_uint64(const char* data_file) {
	std::cout << "------------------------------------ \n";
	std::cout << "-- Init (uint64):  \n";
	cudaDeviceSynchronize();

	std::ifstream file(data_file, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << data_file << std::endl;
		return -1;
	}
	uint64_t file_size = file.tellg();
	file.seekg(0, std::ios::beg);

	const uint64_t warp_sz          = 32;
	const uint64_t vec_sz           = 1024;
	uint64_t       total_elements   = file_size / sizeof(uint64_t);
	const uint64_t n_vec            = (total_elements / vec_sz);
	const uint64_t n_tup            = vec_sz * n_vec;
	const uint64_t v_blc_sz         = 1;
	const uint64_t n_blc            = n_vec / v_blc_sz;
	const uint64_t n_trd            = v_blc_sz * warp_sz;

	std::cout << "-- File: " << data_file << "\n";
	std::cout << "-- File size: " << file_size << " bytes\n";
	std::cout << "-- Total elements: " << total_elements << "\n";
	std::cout << "-- Processing elements: " << n_tup << "\n";
	std::cout << "-- Number of vectors: " << n_vec << "\n";

	auto*          h_org_arr             = new uint64_t[n_tup];
	auto*          h_encoded_data        = new uint64_t[n_tup];
	auto*          h_decoded_arr         = new uint64_t[n_tup];
	auto*          h_transposed_arr      = new uint64_t[vec_sz];
	auto*          h_unrsummed_arr       = new uint64_t[vec_sz];
	auto*          h_base_arr            = new uint64_t[16 * n_vec];  // 16 chains for uint64 rsum
	auto*          h_expected_transposed = new uint64_t[n_tup];
	auto*          h_expected_original   = new uint64_t[n_tup];  // 原始顺序期望值
	uint64_t*      d_base_arr            = nullptr;
	uint64_t*      d_decoded_arr         = nullptr;
	uint64_t*      d_encoded_arr         = nullptr;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint64_t) * n_tup));

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Load data from file : \n";

	file.read(reinterpret_cast<char*>(h_org_arr), n_tup * sizeof(uint64_t));
	file.close();

	uint64_t min_val = h_org_arr[0], max_val = h_org_arr[0];
	for (uint64_t i = 1; i < std::min(n_tup, (uint64_t)1000000); i++) {
		if (h_org_arr[i] < min_val) min_val = h_org_arr[i];
		if (h_org_arr[i] > max_val) max_val = h_org_arr[i];
	}
	std::cout << "-- Data stats (first 1M): min=" << min_val << ", max=" << max_val << "\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Compute optimal bitwidth : \n";

	uint8_t num_bits = compute_max_delta_bitwidth_64(h_transposed_arr, h_unrsummed_arr,
	                                                  h_org_arr, n_vec, vec_sz);
	std::cout << "-- Computed delta bitwidth: " << (int)num_bits << " bits\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Encode :  \n";

	auto in_als      = h_org_arr;
	auto out_als     = h_encoded_data;
	auto base_als    = h_base_arr;
	auto exp_trans   = h_expected_transposed;
	auto exp_orig    = h_expected_original;

	uint64_t original_size = n_tup * sizeof(uint64_t);
	auto encode_start = std::chrono::high_resolution_clock::now();

	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_als, h_transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(h_transposed_arr, h_unrsummed_arr);

		// Save base values (first 16 values for 16 rsum chains in uint64)
		std::memcpy(base_als, h_transposed_arr, sizeof(uint64_t) * 16);

		// Save transposed expected values
		std::memcpy(exp_trans, h_transposed_arr, sizeof(uint64_t) * vec_sz);

		// Save original order expected values
		std::memcpy(exp_orig, in_als, sizeof(uint64_t) * vec_sz);

		// Pack delta values
		generated::pack::fallback::scalar::pack(h_unrsummed_arr, out_als, num_bits);

		in_als     = in_als + vec_sz;
		out_als    = out_als + (num_bits * vec_sz / 64);
		base_als   = base_als + 16;
		exp_trans  = exp_trans + vec_sz;
		exp_orig   = exp_orig + vec_sz;
	}
	auto encode_end = std::chrono::high_resolution_clock::now();
	double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
	double encode_throughput_gbps = (original_size / 1e9) / (encode_time_ms / 1000.0);

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Load encoded data into GPU : \n";

	uint64_t encoded_size = n_vec * (num_bits * vec_sz / 64) * sizeof(uint64_t);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_encoded_arr, encoded_size));
	CUDA_SAFE_CALL(cudaMemcpy(d_encoded_arr, h_encoded_data, encoded_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_base_arr, 16 * n_vec * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMemcpy(d_base_arr, h_base_arr, 16 * n_vec * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	uint64_t compressed_data_size = encoded_size;
	uint64_t base_size = 16 * n_vec * sizeof(uint64_t);
	uint64_t total_compressed_size = compressed_data_size + base_size;
	double compression_ratio = (double)original_size / total_compressed_size;

	std::cout << "-- Original size: " << (original_size / 1024.0 / 1024.0) << " MB\n";
	std::cout << "-- Compressed data size: " << (compressed_data_size / 1024.0 / 1024.0) << " MB\n";
	std::cout << "-- Base array size: " << (base_size / 1024.0 / 1024.0) << " MB\n";
	std::cout << "-- Compression ratio: " << compression_ratio << "x\n";
	std::cout << "-- Encode time: " << encode_time_ms << " ms\n";
	std::cout << "-- Encode throughput: " << encode_throughput_gbps << " GB/s\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';

	// Test: Delta no transpose (output in transposed order)
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - NO inverse transpose (uint64) : \n";

	const int num_runs = 10;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warmup
	delta_decode_no_transpose_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_no_transpose_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_no_transpose = 0;
	cudaEventElapsedTime(&decode_time_no_transpose, start, stop);
	decode_time_no_transpose /= num_runs;

	double throughput_gbps = (original_size / 1e9) / (decode_time_no_transpose / 1000.0);
	double throughput_geps = (n_tup / 1e9) / (decode_time_no_transpose / 1000.0);

	std::cout << "-- Decode time (avg of " << num_runs << " runs): " << decode_time_no_transpose << " ms\n";
	std::cout << "-- Decode throughput: " << throughput_gbps << " GB/s\n";
	std::cout << "-- Decode throughput: " << throughput_geps << " G elements/s\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy & Verify (NO inverse transpose) :  \n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	uint64_t errors = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_transposed[i] != h_decoded_arr[i]) {
			errors++;
			if (errors <= 5) {
				std::cout << "ERROR: idx " << i << " : " << h_expected_transposed[i] << " != " << h_decoded_arr[i] << "\n";
			}
		}
	}

	if (errors > 0) {
		std::cout << fastlanes::debug::red << "-- FAILED: " << errors << " errors out of "
		          << n_tup << " elements" << fastlanes::debug::def << '\n';
	} else {
		std::cout << fastlanes::debug::green << "-- All " << n_tup << " elements verified correctly!"
		          << fastlanes::debug::def << '\n';
	}

	// ===== Test: Delta with optimized transpose (output in original order) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - OPTIMIZED inverse transpose (uint64) : \n";

	// Warmup
	delta_decode_optimized_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_optimized_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_optimized = 0;
	cudaEventElapsedTime(&decode_time_optimized, start, stop);
	decode_time_optimized /= num_runs;

	double throughput_opt_gbps = (original_size / 1e9) / (decode_time_optimized / 1000.0);
	double throughput_opt_geps = (n_tup / 1e9) / (decode_time_optimized / 1000.0);

	std::cout << "-- Decode time (avg of " << num_runs << " runs): " << decode_time_optimized << " ms\n";
	std::cout << "-- Decode throughput: " << throughput_opt_gbps << " GB/s\n";
	std::cout << "-- Decode throughput: " << throughput_opt_geps << " G elements/s\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy & Verify (OPTIMIZED inverse transpose) :  \n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	uint64_t errors_optimized = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) {
			errors_optimized++;
			if (errors_optimized <= 5) {
				std::cout << "ERROR: idx " << i << " : " << h_expected_original[i] << " != " << h_decoded_arr[i] << "\n";
			}
		}
	}

	if (errors_optimized > 0) {
		std::cout << fastlanes::debug::red << "-- FAILED: " << errors_optimized << " errors out of "
		          << n_tup << " elements" << fastlanes::debug::def << '\n';
	} else {
		std::cout << fastlanes::debug::green << "-- All " << n_tup << " elements verified correctly!"
		          << fastlanes::debug::def << '\n';
	}

	// ===== Test: COMPUTE ONLY (uint64) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - COMPUTE ONLY (uint64, no padding) : \n";

	delta_decode_compute_only_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_compute_only_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_compute_only_64 = 0;
	cudaEventElapsedTime(&decode_time_compute_only_64, start, stop);
	decode_time_compute_only_64 /= num_runs;

	double throughput_co64_gbps = (original_size / 1e9) / (decode_time_compute_only_64 / 1000.0);

	std::cout << "-- Decode time (avg of " << num_runs << " runs): " << decode_time_compute_only_64 << " ms\n";
	std::cout << "-- Decode throughput: " << throughput_co64_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_co64 = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) {
			errors_co64++;
		}
	}
	std::cout << "-- Correctness: " << (errors_co64 == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== Test: COMPUTE + PADDED (uint64) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - COMPUTE+PADDED (uint64) : \n";

	delta_decode_compute_padded_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_compute_padded_64<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_compute_padded_64 = 0;
	cudaEventElapsedTime(&decode_time_compute_padded_64, start, stop);
	decode_time_compute_padded_64 /= num_runs;

	double throughput_cp64_gbps = (original_size / 1e9) / (decode_time_compute_padded_64 / 1000.0);

	std::cout << "-- Decode time (avg of " << num_runs << " runs): " << decode_time_compute_padded_64 << " ms\n";
	std::cout << "-- Decode throughput: " << throughput_cp64_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	uint64_t errors_cp64 = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) {
			errors_cp64++;
		}
	}
	std::cout << "-- Correctness: " << (errors_cp64 == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== Per-Block Encoding for uint64 =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Per-Block Encoding (uint64) : \n";

	auto* h_bitwidths_64 = new uint8_t[n_vec];
	auto* h_offsets_64 = new uint64_t[n_vec + 1];

	compute_per_block_bitwidths_64(h_transposed_arr, h_unrsummed_arr,
	                                h_org_arr, n_vec, vec_sz, h_bitwidths_64, h_offsets_64);

	uint64_t total_compressed_64 = h_offsets_64[n_vec];
	auto* h_encoded_pb_64 = new uint64_t[total_compressed_64];

	// Encode per-block
	auto in_pb_64 = h_org_arr;
	auto out_pb_64 = h_encoded_pb_64;
	for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_pb_64, h_transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(h_transposed_arr, h_unrsummed_arr);
		generated::pack::fallback::scalar::pack(h_unrsummed_arr, out_pb_64, h_bitwidths_64[vec_idx]);
		in_pb_64 += vec_sz;
		out_pb_64 += (h_bitwidths_64[vec_idx] * vec_sz / 64);
	}

	double compression_ratio_pb_64 = (double)original_size / (total_compressed_64 * sizeof(uint64_t));
	std::cout << "-- Per-Block compressed size: " << (total_compressed_64 * sizeof(uint64_t) / 1024.0 / 1024.0) << " MB\n";
	std::cout << "-- Per-Block compression ratio: " << compression_ratio_pb_64 << "x\n";

	// Allocate GPU memory for per-block
	uint64_t* d_encoded_pb_64 = nullptr;
	uint8_t* d_bitwidths_64 = nullptr;
	uint64_t* d_offsets_64 = nullptr;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_encoded_pb_64, sizeof(uint64_t) * total_compressed_64));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_bitwidths_64, sizeof(uint8_t) * n_vec));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_offsets_64, sizeof(uint64_t) * (n_vec + 1)));

	CUDA_SAFE_CALL(cudaMemcpy(d_encoded_pb_64, h_encoded_pb_64, sizeof(uint64_t) * total_compressed_64, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_bitwidths_64, h_bitwidths_64, sizeof(uint8_t) * n_vec, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_offsets_64, h_offsets_64, sizeof(uint64_t) * (n_vec + 1), cudaMemcpyHostToDevice));

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';

	// ===== Per-Block No Transpose (uint64) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block, NO transpose (uint64) : \n";

	delta_decode_per_block_no_transpose_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_no_transpose_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_no_64 = 0;
	cudaEventElapsedTime(&decode_time_pb_no_64, start, stop);
	decode_time_pb_no_64 /= num_runs;
	double throughput_pb_no_64 = (original_size / 1e9) / (decode_time_pb_no_64 / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_no_64 << " ms, Throughput: " << throughput_pb_no_64 << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_no_64 = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_transposed[i] != h_decoded_arr[i]) errors_pb_no_64++;
	}
	std::cout << "-- Correctness: " << (errors_pb_no_64 == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== Per-Block Optimized (uint64) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block, OPTIMIZED (uint64) : \n";

	delta_decode_per_block_optimized_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_optimized_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_opt_64 = 0;
	cudaEventElapsedTime(&decode_time_pb_opt_64, start, stop);
	decode_time_pb_opt_64 /= num_runs;
	double throughput_pb_opt_64 = (original_size / 1e9) / (decode_time_pb_opt_64 / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_opt_64 << " ms, Throughput: " << throughput_pb_opt_64 << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_opt_64 = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_pb_opt_64++;
	}
	std::cout << "-- Correctness: " << (errors_pb_opt_64 == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== Per-Block COMPUTE ONLY (uint64) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block, COMPUTE ONLY (uint64) : \n";

	delta_decode_per_block_compute_only_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_compute_only_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_co_64 = 0;
	cudaEventElapsedTime(&decode_time_pb_co_64, start, stop);
	decode_time_pb_co_64 /= num_runs;
	double throughput_pb_co_64 = (original_size / 1e9) / (decode_time_pb_co_64 / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_co_64 << " ms, Throughput: " << throughput_pb_co_64 << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_co_64 = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_pb_co_64++;
	}
	std::cout << "-- Correctness: " << (errors_pb_co_64 == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== Per-Block COMPUTE+PADDED (uint64) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block, COMPUTE+PADDED (uint64) : \n";

	delta_decode_per_block_compute_padded_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_compute_padded_64<<<n_blc, n_trd>>>(d_encoded_pb_64, d_decoded_arr, d_base_arr, d_bitwidths_64, d_offsets_64);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_cp_64 = 0;
	cudaEventElapsedTime(&decode_time_pb_cp_64, start, stop);
	decode_time_pb_cp_64 /= num_runs;
	double throughput_pb_cp_64 = (original_size / 1e9) / (decode_time_pb_cp_64 / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_cp_64 << " ms, Throughput: " << throughput_pb_cp_64 << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_cp_64 = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_pb_cp_64++;
	}
	std::cout << "-- Correctness: " << (errors_pb_cp_64 == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== Summary =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Summary (uint64):  \n";
	std::cout << "-- Original size: " << (original_size / 1024.0 / 1024.0) << " MB\n";
	std::cout << "-- [Uniform] Compression ratio: " << compression_ratio << "x\n";
	std::cout << "-- [Per-Block] Compression ratio: " << compression_ratio_pb_64 << "x\n";
	std::cout << "-- \n";
	std::cout << "-- | Version                         | Time (ms) | Throughput (GB/s) |\n";
	std::cout << "-- |---------------------------------|-----------|-------------------|\n";
	std::cout << "-- | Uniform, No transpose           | " << decode_time_no_transpose << " | " << throughput_gbps << " |\n";
	std::cout << "-- | Uniform, Optimized              | " << decode_time_optimized << " | " << throughput_opt_gbps << " |\n";
	std::cout << "-- | Uniform, COMPUTE ONLY           | " << decode_time_compute_only_64 << " | " << throughput_co64_gbps << " |\n";
	std::cout << "-- | Uniform, COMPUTE+PADDED         | " << decode_time_compute_padded_64 << " | " << throughput_cp64_gbps << " |\n";
	std::cout << "-- | Per-Block, No transpose         | " << decode_time_pb_no_64 << " | " << throughput_pb_no_64 << " |\n";
	std::cout << "-- | Per-Block, Optimized            | " << decode_time_pb_opt_64 << " | " << throughput_pb_opt_64 << " |\n";
	std::cout << "-- | Per-Block, COMPUTE ONLY         | " << decode_time_pb_co_64 << " | " << throughput_pb_co_64 << " |\n";
	std::cout << "-- | Per-Block, COMPUTE+PADDED       | " << decode_time_pb_cp_64 << " | " << throughput_pb_cp_64 << " |\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	delete[] h_org_arr;
	delete[] h_encoded_data;
	delete[] h_decoded_arr;
	delete[] h_transposed_arr;
	delete[] h_unrsummed_arr;
	delete[] h_base_arr;
	delete[] h_expected_transposed;
	delete[] h_expected_original;
	delete[] h_bitwidths_64;
	delete[] h_offsets_64;
	delete[] h_encoded_pb_64;

	CUDA_SAFE_CALL(cudaFree(d_decoded_arr));
	CUDA_SAFE_CALL(cudaFree(d_encoded_arr));
	CUDA_SAFE_CALL(cudaFree(d_base_arr));
	CUDA_SAFE_CALL(cudaFree(d_encoded_pb_64));
	CUDA_SAFE_CALL(cudaFree(d_bitwidths_64));
	CUDA_SAFE_CALL(cudaFree(d_offsets_64));

	return (errors == 0 && errors_optimized == 0) ? 0 : -1;
}

int run_benchmark_uint32(const char* data_file) {

	/* Init */
	std::cout << "------------------------------------ \n";
	std::cout << "-- Init :  \n";
	cudaDeviceSynchronize();

	// 读取文件大小
	std::ifstream file(data_file, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << data_file << std::endl;
		return -1;
	}
	uint64_t file_size = file.tellg();
	file.seekg(0, std::ios::beg);

	const uint64_t warp_sz          = 32;
	const uint64_t vec_sz           = 1024;
	uint64_t       total_elements   = file_size / sizeof(uint32_t);
	const uint64_t n_vec            = (total_elements / vec_sz);
	const uint64_t n_tup            = vec_sz * n_vec;
	const uint64_t v_blc_sz         = 1;
	const uint64_t n_blc            = n_vec / v_blc_sz;
	const uint64_t n_trd            = v_blc_sz * warp_sz;

	std::cout << "-- File: " << data_file << "\n";
	std::cout << "-- File size: " << file_size << " bytes\n";
	std::cout << "-- Total elements: " << total_elements << "\n";
	std::cout << "-- Processing elements: " << n_tup << "\n";
	std::cout << "-- Number of vectors: " << n_vec << "\n";

	auto*          h_org_arr             = new uint32_t[n_tup];
	auto*          h_encoded_data        = new uint32_t[n_tup];
	auto*          h_encoded_data_pb     = new uint32_t[n_tup];  // per-block编码数据
	auto*          h_decoded_arr         = new uint32_t[n_tup];
	auto*          h_transposed_arr      = new uint32_t[vec_sz];
	auto*          h_unrsummed_arr       = new uint32_t[vec_sz];
	auto*          h_base_arr            = new uint32_t[32 * n_vec];
	auto*          h_bitwidths           = new uint8_t[n_vec];    // per-block位宽
	auto*          h_offsets             = new uint64_t[n_vec + 1]; // per-block偏移
	auto*          h_expected_original   = new uint32_t[n_tup];  // 原始顺序期望值
	auto*          h_expected_transposed = new uint32_t[n_tup];  // 转置顺序期望值
	uint32_t*      d_base_arr            = nullptr;
	uint32_t*      d_decoded_arr         = nullptr;
	uint32_t*      d_encoded_arr         = nullptr;
	uint32_t*      d_encoded_arr_pb      = nullptr;  // per-block编码GPU数据
	uint8_t*       d_bitwidths           = nullptr;
	uint64_t*      d_offsets             = nullptr;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint32_t) * n_tup));

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Load data from file : \n";

	/* 从文件读取数据（不排序） */
	file.read(reinterpret_cast<char*>(h_org_arr), n_tup * sizeof(uint32_t));
	file.close();

	// 打印数据统计信息
	uint32_t min_val = h_org_arr[0], max_val = h_org_arr[0];
	for (uint64_t i = 1; i < std::min(n_tup, (uint64_t)1000000); i++) {
		if (h_org_arr[i] < min_val) min_val = h_org_arr[i];
		if (h_org_arr[i] > max_val) max_val = h_org_arr[i];
	}
	std::cout << "-- Data stats (first 1M): min=" << min_val << ", max=" << max_val << "\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Compute optimal bitwidth : \n";

	// 计算全局统一位宽（遍历所有向量找最大delta）
	uint8_t num_bits = compute_max_delta_bitwidth(h_transposed_arr, h_unrsummed_arr,
	                                               h_org_arr, n_vec, vec_sz);
	std::cout << "-- Global uniform bitwidth: " << (int)num_bits << " bits\n";

	// 计算per-block位宽
	compute_per_block_bitwidths(h_transposed_arr, h_unrsummed_arr,
	                             h_org_arr, n_vec, vec_sz, h_bitwidths, h_offsets);

	// 统计per-block位宽分布
	uint64_t bw_histogram[33] = {0};
	for (uint64_t i = 0; i < n_vec; i++) {
		bw_histogram[h_bitwidths[i]]++;
	}
	std::cout << "-- Per-block bitwidth distribution:\n";
	for (int bw = 1; bw <= 32; bw++) {
		if (bw_histogram[bw] > 0) {
			std::cout << "   bw=" << bw << ": " << bw_histogram[bw] << " blocks ("
			          << (100.0 * bw_histogram[bw] / n_vec) << "%)\n";
		}
	}

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Encode (uniform bitwidth) :  \n";

	auto in_als      = h_org_arr;
	auto out_als     = h_encoded_data;
	auto base_als    = h_base_arr;
	auto exp_orig    = h_expected_original;
	auto exp_trans   = h_expected_transposed;

	uint64_t original_size = n_tup * sizeof(uint32_t);

	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_als, h_transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(h_transposed_arr, h_unrsummed_arr);

		// 保存base（每个向量的32个基准值）
		std::memcpy(base_als, h_transposed_arr, sizeof(uint32_t) * 32);

		// 保存原始顺序期望值（用于验证含逆转置版本）
		std::memcpy(exp_orig, in_als, sizeof(uint32_t) * vec_sz);

		// 保存转置顺序期望值（用于验证无逆转置版本）
		std::memcpy(exp_trans, h_transposed_arr, sizeof(uint32_t) * vec_sz);

		// Pack delta values (uniform bitwidth)
		generated::pack::fallback::scalar::pack(h_unrsummed_arr, out_als, num_bits);

		in_als     = in_als + vec_sz;
		out_als    = out_als + (num_bits * vec_sz / 32);
		base_als   = base_als + 32;
		exp_orig   = exp_orig + vec_sz;
		exp_trans  = exp_trans + vec_sz;
	}

	std::cout << "------------------------------------ \n";
	std::cout << "-- Encode (per-block bitwidth) :  \n";

	// Per-block编码
	in_als = h_org_arr;
	out_als = h_encoded_data_pb;
	auto encode_start_pb = std::chrono::high_resolution_clock::now();
	for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
		generated::transpose::fallback::scalar::transpose_i(in_als, h_transposed_arr);
		generated::unrsum::fallback::scalar::unrsum(h_transposed_arr, h_unrsummed_arr);

		// Pack with per-block bitwidth
		uint8_t bw = h_bitwidths[vec_idx];
		generated::pack::fallback::scalar::pack(h_unrsummed_arr, out_als, bw);

		in_als  = in_als + vec_sz;
		out_als = out_als + (bw * vec_sz / 32);
	}
	auto encode_end_pb = std::chrono::high_resolution_clock::now();
	double encode_time_ms_pb = std::chrono::duration<double, std::milli>(encode_end_pb - encode_start_pb).count();
	double encode_throughput_gbps_pb = (original_size / 1e9) / (encode_time_ms_pb / 1000.0);

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Load encoded data into GPU : \n";

	// Uniform bitwidth
	uint64_t encoded_size = n_vec * (num_bits * vec_sz / 32) * sizeof(uint32_t);
	d_encoded_arr = fastlanes::gpu::load_arr(h_encoded_data, encoded_size);
	d_base_arr    = fastlanes::gpu::load_arr(h_base_arr, 32 * n_vec * sizeof(uint32_t));

	// Per-block bitwidth
	uint64_t encoded_size_pb = h_offsets[n_vec] * sizeof(uint32_t);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_encoded_arr_pb, encoded_size_pb));
	CUDA_SAFE_CALL(cudaMemcpy(d_encoded_arr_pb, h_encoded_data_pb, encoded_size_pb, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_bitwidths, n_vec * sizeof(uint8_t)));
	CUDA_SAFE_CALL(cudaMemcpy(d_bitwidths, h_bitwidths, n_vec * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_offsets, (n_vec + 1) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMemcpy(d_offsets, h_offsets, (n_vec + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 计算压缩率
	uint64_t compressed_data_size = encoded_size;
	uint64_t compressed_data_size_pb = encoded_size_pb;
	uint64_t base_size = 32 * n_vec * sizeof(uint32_t);
	uint64_t metadata_size_pb = n_vec * sizeof(uint8_t) + (n_vec + 1) * sizeof(uint64_t);
	uint64_t total_compressed_size = compressed_data_size + base_size;
	uint64_t total_compressed_size_pb = compressed_data_size_pb + base_size + metadata_size_pb;
	double compression_ratio = (double)original_size / total_compressed_size;
	double compression_ratio_pb = (double)original_size / total_compressed_size_pb;

	std::cout << "-- Original size: " << (original_size / 1024.0 / 1024.0) << " MB\n";
	std::cout << "-- [Uniform] Compressed data: " << (compressed_data_size / 1024.0 / 1024.0) << " MB, "
	          << "Total: " << (total_compressed_size / 1024.0 / 1024.0) << " MB, "
	          << "Ratio: " << compression_ratio << "x\n";
	std::cout << "-- [Per-Block] Compressed data: " << (compressed_data_size_pb / 1024.0 / 1024.0) << " MB, "
	          << "Metadata: " << (metadata_size_pb / 1024.0 / 1024.0) << " MB, "
	          << "Total: " << (total_compressed_size_pb / 1024.0 / 1024.0) << " MB, "
	          << "Ratio: " << compression_ratio_pb << "x\n";
	std::cout << "-- [Per-Block] Encode time: " << encode_time_ms_pb << " ms\n";
	std::cout << "-- [Per-Block] Encode throughput: " << encode_throughput_gbps_pb << " GB/s\n";
	std::cout << "-- Per-block improvement: " << ((compression_ratio_pb / compression_ratio - 1) * 100) << "%\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';

	// ===== 测试1: Delta无逆转置（输出转置顺序）=====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Uniform BW, NO inverse transpose : \n";

	const int num_runs = 10;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// 预热
	delta_decode_no_transpose<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 测量解码时间
	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_no_transpose<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_no_transpose = 0;
	cudaEventElapsedTime(&decode_time_no_transpose, start, stop);
	decode_time_no_transpose /= num_runs;

	double throughput_no_transpose_gbps = (original_size / 1e9) / (decode_time_no_transpose / 1000.0);
	double throughput_no_transpose_geps = (n_tup / 1e9) / (decode_time_no_transpose / 1000.0);

	std::cout << "-- Decode time (avg of " << num_runs << " runs): " << decode_time_no_transpose << " ms\n";
	std::cout << "-- Decode throughput: " << throughput_no_transpose_gbps << " GB/s\n";
	std::cout << "-- Decode throughput: " << throughput_no_transpose_geps << " G elements/s\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy & Verify (NO inverse transpose) :  \n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 验证正确性（与转置顺序期望值对比）
	uint64_t errors_no_transpose = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_transposed[i] != h_decoded_arr[i]) {
			errors_no_transpose++;
		}
	}

	if (errors_no_transpose > 0) {
		std::cout << fastlanes::debug::red << "-- FAILED: " << errors_no_transpose << " errors out of "
		          << n_tup << " elements" << fastlanes::debug::def << '\n';
	} else {
		std::cout << fastlanes::debug::green << "-- All " << n_tup << " elements verified correctly!"
		          << fastlanes::debug::def << '\n';
	}

	// ===== 测试2: Delta含逆转置（输出原始顺序）=====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - WITH inverse transpose : \n";

	// 预热
	delta_decode_with_transpose<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 测量解码时间
	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_with_transpose<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_with_transpose = 0;
	cudaEventElapsedTime(&decode_time_with_transpose, start, stop);
	decode_time_with_transpose /= num_runs;

	double throughput_with_transpose_gbps = (original_size / 1e9) / (decode_time_with_transpose / 1000.0);
	double throughput_with_transpose_geps = (n_tup / 1e9) / (decode_time_with_transpose / 1000.0);

	std::cout << "-- Decode time (avg of " << num_runs << " runs): " << decode_time_with_transpose << " ms\n";
	std::cout << "-- Decode throughput: " << throughput_with_transpose_gbps << " GB/s\n";
	std::cout << "-- Decode throughput: " << throughput_with_transpose_geps << " G elements/s\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy & Verify (WITH inverse transpose) :  \n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 验证正确性（与原始顺序期望值对比）
	uint64_t errors_with_transpose = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) {
			errors_with_transpose++;
		}
	}

	if (errors_with_transpose > 0) {
		std::cout << fastlanes::debug::red << "-- FAILED: " << errors_with_transpose << " errors out of "
		          << n_tup << " elements" << fastlanes::debug::def << '\n';
	} else {
		std::cout << fastlanes::debug::green << "-- All " << n_tup << " elements verified correctly!"
		          << fastlanes::debug::def << '\n';
	}

	// ===== 测试3: Delta优化版逆转置（顺序写全局内存）=====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - OPTIMIZED inverse transpose : \n";

	// 预热
	delta_decode_optimized<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 测量解码时间
	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_optimized<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_optimized = 0;
	cudaEventElapsedTime(&decode_time_optimized, start, stop);
	decode_time_optimized /= num_runs;

	double throughput_optimized_gbps = (original_size / 1e9) / (decode_time_optimized / 1000.0);
	double throughput_optimized_geps = (n_tup / 1e9) / (decode_time_optimized / 1000.0);

	std::cout << "-- Decode time (avg of " << num_runs << " runs): " << decode_time_optimized << " ms\n";
	std::cout << "-- Decode throughput: " << throughput_optimized_gbps << " GB/s\n";
	std::cout << "-- Decode throughput: " << throughput_optimized_geps << " G elements/s\n";

	std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
	std::cout << "------------------------------------ \n";
	std::cout << "-- Copy & Verify (OPTIMIZED inverse transpose) :  \n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// 验证正确性（与原始顺序期望值对比）
	uint64_t errors_optimized = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) {
			errors_optimized++;
		}
	}

	if (errors_optimized > 0) {
		std::cout << fastlanes::debug::red << "-- FAILED: " << errors_optimized << " errors out of "
		          << n_tup << " elements" << fastlanes::debug::def << '\n';
	} else {
		std::cout << fastlanes::debug::green << "-- All " << n_tup << " elements verified correctly!"
		          << fastlanes::debug::def << '\n';
	}

	// ===== 测试4: 计算公式版本 =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - COMPUTE formula : \n";

	delta_decode_compute<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_compute<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_compute = 0;
	cudaEventElapsedTime(&decode_time_compute, start, stop);
	decode_time_compute /= num_runs;

	double throughput_compute_gbps = (original_size / 1e9) / (decode_time_compute / 1000.0);
	std::cout << "-- Decode time: " << decode_time_compute << " ms, Throughput: " << throughput_compute_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_compute = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_compute++;
	}
	std::cout << "-- Correctness: " << (errors_compute == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试5: Padding版本 =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - PADDED shared memory : \n";

	delta_decode_padded<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_padded<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_padded = 0;
	cudaEventElapsedTime(&decode_time_padded, start, stop);
	decode_time_padded /= num_runs;

	double throughput_padded_gbps = (original_size / 1e9) / (decode_time_padded / 1000.0);
	std::cout << "-- Decode time: " << decode_time_padded << " ms, Throughput: " << throughput_padded_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_padded = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_padded++;
	}
	std::cout << "-- Correctness: " << (errors_padded == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试6: 计算+Padding组合版本 =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - COMPUTE + PADDED : \n";

	delta_decode_compute_padded<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_compute_padded<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_compute_padded = 0;
	cudaEventElapsedTime(&decode_time_compute_padded, start, stop);
	decode_time_compute_padded /= num_runs;

	double throughput_compute_padded_gbps = (original_size / 1e9) / (decode_time_compute_padded / 1000.0);
	std::cout << "-- Decode time: " << decode_time_compute_padded << " ms, Throughput: " << throughput_compute_padded_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_compute_padded = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_compute_padded++;
	}
	std::cout << "-- Correctness: " << (errors_compute_padded == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试7: 融合版本（rsum直接写全局内存）=====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - FUSED (rsum direct write) : \n";

	delta_decode_fused<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_fused<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_fused = 0;
	cudaEventElapsedTime(&decode_time_fused, start, stop);
	decode_time_fused /= num_runs;

	double throughput_fused_gbps = (original_size / 1e9) / (decode_time_fused / 1000.0);
	std::cout << "-- Decode time: " << decode_time_fused << " ms, Throughput: " << throughput_fused_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_fused = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_fused++;
	}
	std::cout << "-- Correctness: " << (errors_fused == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试8: 寄存器缓冲版本 =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - REG BUFFER : \n";

	delta_decode_reg_buffer<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_reg_buffer<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_reg = 0;
	cudaEventElapsedTime(&decode_time_reg, start, stop);
	decode_time_reg /= num_runs;

	double throughput_reg_gbps = (original_size / 1e9) / (decode_time_reg / 1000.0);
	std::cout << "-- Decode time: " << decode_time_reg << " ms, Throughput: " << throughput_reg_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_reg = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_reg++;
	}
	std::cout << "-- Correctness: " << (errors_reg == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试9: Warp协作写入版本 =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - WARP WRITE : \n";

	delta_decode_warp_write<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_warp_write<<<n_blc, n_trd>>>(d_encoded_arr, d_decoded_arr, d_base_arr, num_bits);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_warp = 0;
	cudaEventElapsedTime(&decode_time_warp, start, stop);
	decode_time_warp /= num_runs;

	double throughput_warp_gbps = (original_size / 1e9) / (decode_time_warp / 1000.0);
	std::cout << "-- Decode time: " << decode_time_warp << " ms, Throughput: " << throughput_warp_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_warp = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_warp++;
	}
	std::cout << "-- Correctness: " << (errors_warp == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试10: Per-block位宽（无逆转置）=====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block, NO transpose : \n";

	delta_decode_per_block_no_transpose<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_no_transpose<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_no_transpose = 0;
	cudaEventElapsedTime(&decode_time_pb_no_transpose, start, stop);
	decode_time_pb_no_transpose /= num_runs;

	double throughput_pb_no_transpose_gbps = (original_size / 1e9) / (decode_time_pb_no_transpose / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_no_transpose << " ms, Throughput: " << throughput_pb_no_transpose_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_no_transpose = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_transposed[i] != h_decoded_arr[i]) errors_pb_no_transpose++;
	}
	std::cout << "-- Correctness: " << (errors_pb_no_transpose == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试11: Per-block位宽（含逆转置，优化版）=====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block BW, OPTIMIZED transpose : \n";

	delta_decode_per_block_optimized<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_optimized<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_optimized = 0;
	cudaEventElapsedTime(&decode_time_pb_optimized, start, stop);
	decode_time_pb_optimized /= num_runs;

	double throughput_pb_optimized_gbps = (original_size / 1e9) / (decode_time_pb_optimized / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_optimized << " ms, Throughput: " << throughput_pb_optimized_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_optimized = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_pb_optimized++;
	}
	std::cout << "-- Correctness: " << (errors_pb_optimized == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试12: Per-block COMPUTE ONLY (无padding) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block, COMPUTE ONLY (no padding) : \n";

	delta_decode_per_block_compute_only<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_compute_only<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_compute_only = 0;
	cudaEventElapsedTime(&decode_time_pb_compute_only, start, stop);
	decode_time_pb_compute_only /= num_runs;

	double throughput_pb_compute_only_gbps = (original_size / 1e9) / (decode_time_pb_compute_only / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_compute_only << " ms, Throughput: " << throughput_pb_compute_only_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_compute_only = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_pb_compute_only++;
	}
	std::cout << "-- Correctness: " << (errors_pb_compute_only == 0 ? "PASSED" : "FAILED") << "\n";

	// ===== 测试13: Per-block位宽 COMPUTE + PADDED (最优化) =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Decode (GPU) - Per-Block, COMPUTE+PADDED : \n";

	delta_decode_per_block_compute_padded<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	cudaEventRecord(start);
	for (int run = 0; run < num_runs; run++) {
		delta_decode_per_block_compute_padded<<<n_blc, n_trd>>>(d_encoded_arr_pb, d_decoded_arr, d_base_arr, d_bitwidths, d_offsets);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float decode_time_pb_compute_padded = 0;
	cudaEventElapsedTime(&decode_time_pb_compute_padded, start, stop);
	decode_time_pb_compute_padded /= num_runs;

	double throughput_pb_compute_padded_gbps = (original_size / 1e9) / (decode_time_pb_compute_padded / 1000.0);
	std::cout << "-- Decode time: " << decode_time_pb_compute_padded << " ms, Throughput: " << throughput_pb_compute_padded_gbps << " GB/s\n";

	CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
	uint64_t errors_pb_compute_padded = 0;
	for (uint64_t i = 0; i < n_tup; i++) {
		if (h_expected_original[i] != h_decoded_arr[i]) errors_pb_compute_padded++;
	}
	std::cout << "-- Correctness: " << (errors_pb_compute_padded == 0 ? "PASSED" : "FAILED") << "\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// ===== 汇总 =====
	std::cout << "------------------------------------ \n";
	std::cout << "-- Summary :  \n";
	std::cout << "-- Elements: " << n_tup << " (" << (original_size / 1024.0 / 1024.0) << " MB)\n";
	std::cout << "-- [Uniform] Bitwidth: " << (int)num_bits << " bits, Compression: " << compression_ratio << "x\n";
	std::cout << "-- [Per-Block] Compression: " << compression_ratio_pb << "x (+" << ((compression_ratio_pb / compression_ratio - 1) * 100) << "%)\n";
	std::cout << "-- \n";
	std::cout << "-- Performance Comparison:\n";
	std::cout << "-- | Version                       | Time (ms) | Throughput (GB/s) |\n";
	std::cout << "-- |-------------------------------|-----------|-------------------|\n";
	std::cout << "-- | Uniform, No transpose         | " << decode_time_no_transpose << " | " << throughput_no_transpose_gbps << " |\n";
	std::cout << "-- | Uniform, Optimized transpose  | " << decode_time_optimized << " | " << throughput_optimized_gbps << " |\n";
	std::cout << "-- | Per-Block, No transpose       | " << decode_time_pb_no_transpose << " | " << throughput_pb_no_transpose_gbps << " |\n";
	std::cout << "-- | Per-Block, Optimized transpose| " << decode_time_pb_optimized << " | " << throughput_pb_optimized_gbps << " |\n";

	// 清理
	delete[] h_org_arr;
	delete[] h_encoded_data;
	delete[] h_encoded_data_pb;
	delete[] h_decoded_arr;
	delete[] h_transposed_arr;
	delete[] h_unrsummed_arr;
	delete[] h_base_arr;
	delete[] h_bitwidths;
	delete[] h_offsets;
	delete[] h_expected_original;
	delete[] h_expected_transposed;

	CUDA_SAFE_CALL(cudaFree(d_decoded_arr));
	CUDA_SAFE_CALL(cudaFree(d_encoded_arr));
	CUDA_SAFE_CALL(cudaFree(d_encoded_arr_pb));
	CUDA_SAFE_CALL(cudaFree(d_base_arr));
	CUDA_SAFE_CALL(cudaFree(d_bitwidths));
	CUDA_SAFE_CALL(cudaFree(d_offsets));

	bool all_passed = (errors_no_transpose == 0 && errors_with_transpose == 0 &&
	                   errors_optimized == 0 && errors_compute == 0 &&
	                   errors_padded == 0 && errors_compute_padded == 0 &&
	                   errors_pb_no_transpose == 0 && errors_pb_optimized == 0);
	return all_passed ? 0 : -1;
}

int main(int argc, char** argv) {
	bool use_uint64 = false;
	const char* data_file = "/home/xiayouyang/code/L3/data/sosd/5-books_200M_uint32.bin";

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--uint64") == 0) {
			use_uint64 = true;
		} else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
			data_file = argv[++i];
		}
	}

	std::cout << "FastLanes Delta Benchmark\n";
	std::cout << "Data type: " << (use_uint64 ? "uint64" : "uint32") << "\n";

	if (use_uint64) {
		return run_benchmark_uint64(data_file);
	} else {
		return run_benchmark_uint32(data_file);
	}
}
