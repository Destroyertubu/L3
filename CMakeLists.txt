cmake_minimum_required(VERSION 3.18)
project(GLECO_OPT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 90)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/kernels/utils)

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
    src/kernels/decompression/decompression_kernels_phase2_bucket.cu
    src/kernels/decompression/decoder_warp_opt.cu
    src/kernels/utils/partition_bounds_kernel.cu
    src/kernels/utils/random_access_kernels.cu
)

# Codec library sources
set(CODEC_SOURCES
    src/codec/L3_codec.cpp
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
    L3_kernels
    L3_codec
    cudart
)

# Original fb_200M test
add_executable(test_fb_200M_phase2_bucket
    src/tests/test_fb_200M_phase2_bucket.cpp
)
target_link_libraries(test_fb_200M_phase2_bucket
    L3_kernels
    L3_codec
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
    L3_kernels
    L3_codec
    cudart
)

# Enable output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Print configuration
message(STATUS "GLECO Optimization Project Configuration:")
message(STATUS "  CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CTA Batch: ${PHASE2_CTA_BATCH}")
message(STATUS "  cp.async: ${PHASE2_USE_CP_ASYNC}")
message(STATUS "  Persistent Threads: ${PHASE2_PERSISTENT_THREADS}")
