# SSB (Star Schema Benchmark) Benchmarks

# Include directories for SSB utilities
include_directories(${CMAKE_SOURCE_DIR}/include/common)

# ============================================================================
# Baseline Benchmarks
# ============================================================================
file(GLOB BASELINE_SOURCES baseline/*.cu)

foreach(source_file ${BASELINE_SOURCES})
    get_filename_component(exec_name ${source_file} NAME_WE)
    add_executable(${exec_name}_baseline ${source_file})

    target_link_libraries(${exec_name}_baseline PRIVATE l32)

    target_compile_options(${exec_name}_baseline PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    set_target_properties(${exec_name}_baseline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ssb/baseline
    )
endforeach()

# ============================================================================
# Optimized 2-Push Benchmarks
# ============================================================================
file(GLOB OPTIMIZED_SOURCES optimized_2push/*.cu)

foreach(source_file ${OPTIMIZED_SOURCES})
    get_filename_component(exec_name ${source_file} NAME_WE)
    add_executable(${exec_name}_opt ${source_file})

    target_link_libraries(${exec_name}_opt PRIVATE l32)

    target_compile_options(${exec_name}_opt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    set_target_properties(${exec_name}_opt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/ssb/optimized
    )
endforeach()

message(STATUS "Configured ${list_length} SSB baseline benchmarks")
message(STATUS "Configured ${list_length} SSB optimized benchmarks")
