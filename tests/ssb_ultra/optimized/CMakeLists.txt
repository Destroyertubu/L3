# SSB Ultra Optimized Implementations
# Phase 1-5 optimizations: Two-Level Fast Hash, Shared Memory Cache, etc.

set(OPTIMIZED_COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/kernels/decompression
)

# Helper function to add optimized query executable
function(add_optimized_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_optimized)
    add_executable(${TARGET_NAME} ${QUERY_NAME}_optimized.cu)
    target_include_directories(${TARGET_NAME} PRIVATE ${OPTIMIZED_COMMON_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE L3_kernels L3_codec)
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# Q1 series - Direct filter queries (no hash joins)
add_optimized_query(q11)
add_optimized_query(q12)
add_optimized_query(q13)

# Q2 series - Two hash joins (supplier, part)
add_optimized_query(q21)
add_optimized_query(q22)
add_optimized_query(q23)

# Q3 series - Two hash joins (customer, supplier)
add_optimized_query(q31)
add_optimized_query(q32)
add_optimized_query(q33)
add_optimized_query(q34)

# Q4 series - Four hash joins (customer, supplier, part, date)
add_optimized_query(q41)
add_optimized_query(q42)
add_optimized_query(q43)

# Ultra-optimized versions with ALL optimizations (Phase 3-5)
function(add_ultra_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_ultra)
    add_executable(${TARGET_NAME} ${QUERY_NAME}_ultra.cu)
    target_include_directories(${TARGET_NAME} PRIVATE ${OPTIMIZED_COMMON_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE L3_kernels L3_codec)
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# Ultra versions - combining all optimizations
add_ultra_query(q41)

message(STATUS "SSB Optimized: Building 13 optimized + 1 ultra SSB queries")
