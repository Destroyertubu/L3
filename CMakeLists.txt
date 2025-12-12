cmake_minimum_required(VERSION 3.18)
project(L3_OPT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 90)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/kernels/utils)
include_directories(${CMAKE_SOURCE_DIR}/src/kernels/compression)

# Common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -fPIC")

# Phase 2 configuration options
option(PHASE2_USE_CP_ASYNC "Enable cp.async pipeline" OFF)
set(PHASE2_CTA_BATCH "4" CACHE STRING "Partitions per CTA")
option(PHASE2_PERSISTENT_THREADS "Enable persistent threads" OFF)
option(PHASE2_DEBUG_ROUTING "Enable debug routing" OFF)
option(PHASE2_DEBUG_VECTORIZATION "Enable vectorization debugging" OFF)

# Add compile definitions
add_compile_definitions(
    PHASE2_USE_CP_ASYNC=$<BOOL:${PHASE2_USE_CP_ASYNC}>
    PHASE2_CTA_BATCH=${PHASE2_CTA_BATCH}
    PHASE2_PERSISTENT_THREADS=$<BOOL:${PHASE2_PERSISTENT_THREADS}>
    PHASE2_DEBUG_ROUTING=$<BOOL:${PHASE2_DEBUG_ROUTING}>
    PHASE2_DEBUG_VECTORIZATION=$<BOOL:${PHASE2_DEBUG_VECTORIZATION}>
)

# Kernel library sources
set(KERNEL_SOURCES
    src/kernels/utils/bitpack_utils.cu
    src/kernels/compression/encoder.cu
    src/kernels/compression/encoder_variable_length.cu
    src/kernels/compression/encoder_variable_length_v2.cu
    src/kernels/compression/encoder_cost_optimal.cu
    src/kernels/compression/encoder_cost_optimal_gpu_merge.cu
    src/kernels/compression/encoder_cost_optimal_gpu_merge_v2.cu
    src/kernels/compression/string_encoder.cu
    src/kernels/decompression/decompression_kernels.cu
    src/kernels/decompression/decompression_kernels_opt.cu
    src/kernels/decompression/decompression_kernels_phase2.cu
    src/kernels/decompression/decompression_kernels_phase2_bucket.cu
    src/kernels/decompression/decoder_warp_opt.cu
    src/kernels/decompression/decoder_specialized.cu
    src/kernels/decompression/decoder_Vertical_opt.cu
    src/kernels/decompression/decoder_partition_opt.cu
    src/kernels/decompression/string_decoder.cu
    src/kernels/compression/encoder_Vertical_opt.cu
    src/kernels/utils/partition_bounds_kernel.cu
    src/kernels/utils/random_access_kernels.cu
)

# Codec library sources
set(CODEC_SOURCES
    src/codec/L3_codec.cpp
    src/codec/L3_string_codec.cpp
)

# Create kernel library
add_library(L3_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(L3_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/kernels/utils
)

# Create codec library
add_library(L3_codec STATIC ${CODEC_SOURCES})
target_include_directories(L3_codec PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Main compression test executable (new clean test)
add_executable(test_compression_main
    src/tests/test_compression_main.cpp
)
target_link_libraries(test_compression_main
    L3_codec
    L3_kernels
    cudart
)

# Original fb_200M test
add_executable(test_fb_200M_phase2_bucket
    src/tests/test_fb_200M_phase2_bucket.cpp
)
target_link_libraries(test_fb_200M_phase2_bucket
    L3_codec
    L3_kernels
    cudart
)

# Random Access comprehensive test
add_executable(test_random_access_comprehensive
    tests/ra/test_random_access_comprehensive.cpp
)
target_link_libraries(test_random_access_comprehensive
    L3_codec
    L3_kernels
    cudart
)

# SSB (Star Schema Benchmark) Query Tests
add_executable(test_ssb_queries
    src/tests/test_ssb_queries.cpp
)
target_link_libraries(test_ssb_queries
    L3_codec
    L3_kernels
    cudart
)

# L3 Internal Comparison Test (Fixed vs V1 vs V2)
add_executable(test_L3_inner_comparison
    src/tests/test_L3_inner_comparison.cpp
)
target_link_libraries(test_L3_inner_comparison
    L3_codec
    L3_kernels
    cudart
)

# Enable output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Print configuration
message(STATUS "L3 Optimization Project Configuration:")
message(STATUS "  CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CTA Batch: ${PHASE2_CTA_BATCH}")
message(STATUS "  cp.async: ${PHASE2_USE_CP_ASYNC}")
message(STATUS "  Persistent Threads: ${PHASE2_PERSISTENT_THREADS}")

# String Compression Test
add_executable(test_string_compression
    src/tests/test_string_compression.cpp
)
target_link_libraries(test_string_compression
    L3_codec
    L3_kernels
    cudart
)

# LeCo Email Dataset Test
add_executable(test_email_dataset
    src/tests/test_email_dataset.cpp
)
target_link_libraries(test_email_dataset
    L3_codec
    L3_kernels
    cudart
)

# GPU Merge Test
add_executable(test_gpu_merge
    tests/test_gpu_merge.cu
)
target_link_libraries(test_gpu_merge
    L3_codec
    L3_kernels
    cudart
)

# GPU Merge Normal Dataset Test
add_executable(test_gpu_merge_normal
    tests/test_gpu_merge_normal.cu
)
target_link_libraries(test_gpu_merge_normal
    L3_codec
    L3_kernels
    cudart
)

# GPU Merge Debug Test
add_executable(test_gpu_merge_debug
    tests/test_gpu_merge_debug.cu
)
target_link_libraries(test_gpu_merge_debug
    L3_codec
    L3_kernels
    cudart
)

# GPU Merge Trace Test
add_executable(test_gpu_merge_trace
    tests/test_gpu_merge_trace.cu
)
target_link_libraries(test_gpu_merge_trace
    L3_codec
    L3_kernels
    cudart
)

# CPU Trace Test
add_executable(test_cpu_trace
    tests/test_cpu_trace.cu
)
target_link_libraries(test_cpu_trace
    L3_codec
    L3_kernels
    cudart
)

# Normal Debug Test
add_executable(test_normal_debug
    tests/test_normal_debug.cu
)
target_link_libraries(test_normal_debug
    L3_codec
    L3_kernels
    cudart
)

# GPU Merge V2 Optimized Test
add_executable(test_gpu_merge_v2
    src/tests/test_gpu_merge_v2.cu
)
target_link_libraries(test_gpu_merge_v2
    L3_codec
    L3_kernels
    cudart
)

# GPU Merge V1 vs V2 Benchmark
add_executable(test_gpu_merge_benchmark
    tests/test_gpu_merge_benchmark.cu
)
target_link_libraries(test_gpu_merge_benchmark
    L3_codec
    L3_kernels
    cudart
)

# V2 All Datasets Test
add_executable(test_v2_all_datasets
    tests/test_v2_all_datasets.cu
)
target_link_libraries(test_v2_all_datasets
    L3_codec
    L3_kernels
    cudart
)

# Encoder Comparison Test (CPU vs GPU vs GPU_ZeroSync)
add_executable(test_encoder_comparison
    tests/test_encoder_comparison.cu
)
target_link_libraries(test_encoder_comparison
    L3_codec
    L3_kernels
    cudart
)

# SOSD Loader Library
add_library(sosd_loader STATIC
    src/tools/sosd_loader.cpp
)
target_include_directories(sosd_loader PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Main Unified Test Program
add_executable(main
    main.cpp
)
target_link_libraries(main
    L3_codec
    L3_kernels
    sosd_loader
    cudart
)

# Vertical vs Horizontal Benchmark
add_executable(benchmark_vertical_horizontal
    tests/benchmark_vertical_horizontal.cu
)
target_link_libraries(benchmark_vertical_horizontal
    L3_codec
    L3_kernels
    sosd_loader
    cudart
)

# Ablation Study Benchmark
add_executable(benchmark_ablation_study
    tests/benchmark_ablation_study.cu
)
target_link_libraries(benchmark_ablation_study
    L3_codec
    L3_kernels
    sosd_loader
    cudart
)

# Model Statistics Test - GPU Adaptive algorithm statistics
add_executable(test_model_statistics
    tests/test_model_statistics.cu
)
target_link_libraries(test_model_statistics
    L3_codec
    L3_kernels
    cudart
)
set_target_properties(test_model_statistics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# SSB Ultra tests - comprehensive SSB benchmark with 3 strategies
add_subdirectory(tests/ssb_ultra)

# Merge Strategy Comparison Benchmark (V2 LINEAR-only vs V3 Multi-Model)
add_executable(benchmark_merge_strategies
    tests/benchmark_merge_strategies.cu
)
target_link_libraries(benchmark_merge_strategies
    L3_codec
    L3_kernels
    cudart
)
set_target_properties(benchmark_merge_strategies PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_ARCHITECTURES "80;86;89;90"
)
