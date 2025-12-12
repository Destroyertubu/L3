# CMakeLists.txt for fused_query_new SSB queries
# Crystal-opt style L3 fused decompression kernels

cmake_minimum_required(VERSION 3.18)

# Include directories
set(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(L3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(L3_SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Common include directories for all targets
set(SSB_FUSED_INCLUDES
    ${COMMON_INCLUDE_DIR}
    ${L3_INCLUDE_DIR}
    ${L3_SRC_DIR}
    ${CMAKE_SOURCE_DIR}/tests/ssb_ultra/common
)

# CUDA flags for optimization
set(SSB_FUSED_CUDA_FLAGS
    -O3
    -use_fast_math
    --expt-relaxed-constexpr
    -lineinfo
)

# Function to add SSB query executable
function(add_ssb_fused_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_fused_new)
    add_executable(${TARGET_NAME} ${QUERY_NAME}.cu)

    target_include_directories(${TARGET_NAME} PRIVATE ${SSB_FUSED_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE L3_codec L3_kernels)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SSB_FUSED_CUDA_FLAGS}>)

    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

# Add Q1.x queries (L3 L3-based)
add_ssb_fused_query(q11)
# add_ssb_fused_query(q12)
# add_ssb_fused_query(q13)

# =============================================================================
# Vertical-style L3 FLS queries (FOR+BitPack only, no L3 dependencies)
# =============================================================================

# Function to add standalone FLS query executable
function(add_ssb_fls_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME})
    set(SOURCE_FILE ${QUERY_NAME}_fls.cu)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        add_executable(${TARGET_NAME} ${SOURCE_FILE})

        target_include_directories(${TARGET_NAME} PRIVATE
            ${COMMON_INCLUDE_DIR}
        )
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SSB_FUSED_CUDA_FLAGS}>)

        set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
    endif()
endfunction()

# Add FLS Q1.x queries (standalone, no L3 dependencies)
add_ssb_fls_query(q11)
add_ssb_fls_query(q12)
add_ssb_fls_query(q13)

# Add FLS Q2.x queries (with hash joins)
add_ssb_fls_query(q21)
add_ssb_fls_query(q22)
add_ssb_fls_query(q23)

# Add FLS Q3.x queries (multi-way hash joins)
add_ssb_fls_query(q31)
add_ssb_fls_query(q32)
add_ssb_fls_query(q33)
add_ssb_fls_query(q34)

# Add FLS Q4.x queries (4-way hash joins)
add_ssb_fls_query(q41)
add_ssb_fls_query(q42)
add_ssb_fls_query(q43)
