# ============================================================================
# L3 Benchmarks - Build Configuration
# ============================================================================

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ============================================================================
# Codec Performance Benchmarks
# ============================================================================
file(GLOB CODEC_BENCH_SOURCES codec/*.cpp)

foreach(bench_file ${CODEC_BENCH_SOURCES})
    get_filename_component(bench_name ${bench_file} NAME_WE)
    add_executable(${bench_name} ${bench_file})

    target_link_libraries(${bench_name} PRIVATE
        l3_modular
        cudart
        cuda
    )

    target_include_directories(${bench_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/modular
        ${CMAKE_SOURCE_DIR}/include/common
        ${CMAKE_SOURCE_DIR}/lib/modular/codec
        ${CMAKE_SOURCE_DIR}/lib/modular/utils
        ${CMAKE_SOURCE_DIR}/lib/modular/data
    )

    set_target_properties(${bench_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/codec_benchmarks
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

message(STATUS "L3 Benchmarks: Codec benchmarks configured")

# ============================================================================
# SSB Database Query Benchmarks
# ============================================================================
add_subdirectory(ssb)

# ============================================================================
# Optional: Future Benchmarks
# ============================================================================
# add_subdirectory(sosd)
# add_subdirectory(random_access)

message(STATUS "L3 Benchmarks: All benchmark suites configured")
