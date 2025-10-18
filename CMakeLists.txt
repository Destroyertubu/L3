cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(L3_Compression
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "L3: GPU-Accelerated Learned Compression Library"
)

# ============================================================================
# Project Options
# ============================================================================

option(BUILD_TESTS "Build unit and integration tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# ============================================================================
# C++ and CUDA Standards
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA Configuration
# ============================================================================

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # Turing, Ampere, Hopper
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

# ============================================================================
# Output Directories
# ============================================================================

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================================
# Dependencies
# ============================================================================

find_package(CUDAToolkit REQUIRED)

# Thrust (header-only, comes with CUDA)
# No explicit find needed, included in CUDA Toolkit

# ============================================================================
# Include Directories
# ============================================================================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ============================================================================
# Subdirectories
# ============================================================================

# Core library modules
add_subdirectory(src/partitioner)
# add_subdirectory(src/core)           # TODO: Migrate
# add_subdirectory(src/compression)    # TODO: Migrate
# add_subdirectory(src/decompression)  # TODO: Migrate
# add_subdirectory(src/random_access)  # TODO: Migrate
# add_subdirectory(src/query)          # TODO: Migrate
# add_subdirectory(src/utils)          # TODO: Migrate

# ============================================================================
# Main Library Target
# ============================================================================

add_library(l3_compression STATIC
    $<TARGET_OBJECTS:l3_partitioner>
    # $<TARGET_OBJECTS:l3_core>           # TODO
    # $<TARGET_OBJECTS:l3_compression_impl>     # TODO
    # $<TARGET_OBJECTS:l3_decompression_impl>   # TODO
    # $<TARGET_OBJECTS:l3_random_access>  # TODO
    # $<TARGET_OBJECTS:l3_utils>          # TODO
)

target_include_directories(l3_compression
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(l3_compression
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)

set_target_properties(l3_compression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Examples
# ============================================================================

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Tests
# ============================================================================

if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)  # TODO: Create tests
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(BUILD_BENCHMARKS)
    # add_subdirectory(benchmarks)  # TODO: Migrate benchmarks
endif()

# ============================================================================
# Python Bindings
# ============================================================================

if(BUILD_PYTHON)
    # add_subdirectory(python)  # TODO: Create Python bindings
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS l3_compression
    EXPORT L3CompressionTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Export targets
install(EXPORT L3CompressionTargets
    FILE L3CompressionTargets.cmake
    NAMESPACE L3::
    DESTINATION lib/cmake/L3Compression
)

# Generate config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/L3CompressionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/L3CompressionConfig.cmake
    INSTALL_DESTINATION lib/cmake/L3Compression
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/L3CompressionConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/L3CompressionConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/L3CompressionConfigVersion.cmake
    DESTINATION lib/cmake/L3Compression
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "L3 Compression Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "  BUILD_BENCHMARKS: ${BUILD_BENCHMARKS}")
message(STATUS "  BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_PYTHON: ${BUILD_PYTHON}")
message(STATUS "  BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "")
