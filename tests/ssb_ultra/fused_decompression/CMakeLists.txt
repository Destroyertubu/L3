# CMakeLists.txt for fused_decompression tests
# True Fused Decompression SSB Queries - All 13 queries

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/kernels/utils
    ${CMAKE_SOURCE_DIR}/src/kernels/decompression
    ${CMAKE_SOURCE_DIR}/tests/ssb_ultra/common
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Function to add true fused query executables
function(add_true_fused_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_true_fused)
    add_executable(${TARGET_NAME} ${QUERY_NAME}_true_fused.cu)
    target_link_libraries(${TARGET_NAME}
        L3_codec
        L3_kernels
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89;90"
    )
endfunction()

# Q1.x Family (no joins, 4 columns)
add_true_fused_query(q11)
add_true_fused_query(q12)
add_true_fused_query(q13)

# Float-optimized version of Q1.1
add_executable(ssb_q11_true_fused_float q11_true_fused_float.cu)
target_link_libraries(ssb_q11_true_fused_float L3_codec L3_kernels)
set_target_properties(ssb_q11_true_fused_float PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# Q2.x Family (3 joins: Date, Part, Supplier)
add_true_fused_query(q21)
add_true_fused_query(q22)
add_true_fused_query(q23)

# Q3.x Family (3 joins: Date, Customer, Supplier)
add_true_fused_query(q31)
add_true_fused_query(q32)
add_true_fused_query(q33)
add_true_fused_query(q34)

# Q4.x Family (3-4 joins: Date, Customer, Supplier, Part)
add_true_fused_query(q41)
add_true_fused_query(q42)
add_true_fused_query(q43)

# =============================================================================
# Fused Optimized (32x32 tile, tile-based metadata)
# =============================================================================

# Q1.1 Fused Optimized - target < 0.8 ms
add_executable(ssb_q11_fused_optimized q11_fused_optimized.cu)
target_link_libraries(ssb_q11_fused_optimized L3_codec L3_kernels)
set_target_properties(ssb_q11_fused_optimized PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# Q1.1 Fused V2 - pre-computed bit offsets
add_executable(ssb_q11_fused_v2 q11_fused_v2.cu)
target_link_libraries(ssb_q11_fused_v2 L3_codec L3_kernels)
set_target_properties(ssb_q11_fused_v2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# Q1.1 Fused V3 - template specialized unpack
add_executable(ssb_q11_fused_v3 q11_fused_v3.cu)
target_link_libraries(ssb_q11_fused_v3 L3_codec L3_kernels)
set_target_properties(ssb_q11_fused_v3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# Q1.1 Fused V4 - Vertical-compatible tile format
add_executable(ssb_q11_fused_v4 q11_fused_v4.cu)
target_include_directories(ssb_q11_fused_v4 PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/VerticalGPU/Vertical/src/include
)
target_link_libraries(ssb_q11_fused_v4 L3_codec L3_kernels)
set_target_properties(ssb_q11_fused_v4 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# Q1.1 Fused V5 - L3 Native Format + Prefetch Strategy
add_executable(ssb_q11_fused_v5 q11_fused_v5.cu)
target_link_libraries(ssb_q11_fused_v5 L3_codec L3_kernels)
set_target_properties(ssb_q11_fused_v5 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# =============================================================================
# V5 Fused Queries - L3 Native Format + Prefetch Strategy (All 13 SSB queries)
# =============================================================================

# Function to add V5 fused query executables
function(add_v5_fused_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_fused_v5)
    add_executable(${TARGET_NAME} ${QUERY_NAME}_fused_v5.cu)
    target_link_libraries(${TARGET_NAME}
        L3_codec
        L3_kernels
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89;90"
    )
endfunction()

# Q1.x Family V5 (simple aggregation)
add_v5_fused_query(q12)
add_v5_fused_query(q13)

# Q2.x Family V5 (brand analysis with joins)
add_v5_fused_query(q21)
add_v5_fused_query(q22)
add_v5_fused_query(q23)

# Q3.x Family V5 (geographic analysis)
add_v5_fused_query(q31)
add_v5_fused_query(q32)
add_v5_fused_query(q33)
add_v5_fused_query(q34)

# Q4.x Family V5 (profit analysis)
add_v5_fused_query(q41)
add_v5_fused_query(q42)
add_v5_fused_query(q43)

# =============================================================================
# V6 Fused Queries - Hash Prefetch + Shared Memory Date HT
# =============================================================================

# Function to add V6 fused query executables
function(add_v6_fused_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_fused_v6)
    add_executable(${TARGET_NAME} v6/${QUERY_NAME}_fused_v6.cu)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/v6
    )
    target_link_libraries(${TARGET_NAME}
        L3_codec
        L3_kernels
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89;90"
    )
endfunction()

# V6 Queries
add_v6_fused_query(q21)

# =============================================================================
# V7 Fused Queries - Crystal-Style Perfect Hash (Vertical Compatible)
# =============================================================================

# Function to add V7 fused query executables
function(add_v7_fused_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_fused_v7)
    add_executable(${TARGET_NAME} v7/${QUERY_NAME}_fused_v7.cu)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/v7
    )
    target_link_libraries(${TARGET_NAME}
        L3_codec
        L3_kernels
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89;90"
    )
endfunction()

# V7 Queries (Crystal-style perfect hash)
add_v7_fused_query(q21)
add_v7_fused_query(q31)
add_v7_fused_query(q41)

# =============================================================================
# V8 Fused Queries - Hierarchical Aggregation + Compact Metadata
# =============================================================================

# Function to add V8 fused query executables
function(add_v8_fused_query QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_fused_v8)
    add_executable(${TARGET_NAME} v8/${QUERY_NAME}_fused_v8.cu)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/v7
        ${CMAKE_CURRENT_SOURCE_DIR}/v8
    )
    target_link_libraries(${TARGET_NAME}
        L3_codec
        L3_kernels
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89;90"
    )
endfunction()

# V8 Queries (Hierarchical aggregation)
add_v8_fused_query(q21)

# V8b variant: Warp key matching instead of shared memory hash
add_executable(ssb_q21_fused_v8b v8/q21_fused_v8b.cu)
target_include_directories(ssb_q21_fused_v8b PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v7
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
)
target_link_libraries(ssb_q21_fused_v8b L3_codec L3_kernels)
set_target_properties(ssb_q21_fused_v8b PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# =============================================================================
# V9 Fused Queries - Vertical-Compatible Architecture
# =============================================================================

add_executable(ssb_q21_fused_v9 v9/q21_fused_v9.cu)
target_include_directories(ssb_q21_fused_v9 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v7
)
target_link_libraries(ssb_q21_fused_v9 L3_codec L3_kernels)
set_target_properties(ssb_q21_fused_v9 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# =============================================================================
# V10 Fused Queries - Template-Specialized Unpack + Compact Metadata
# =============================================================================

add_executable(ssb_q21_fused_v10 v10/q21_fused_v10.cu)
target_include_directories(ssb_q21_fused_v10 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v7
)
target_link_libraries(ssb_q21_fused_v10 L3_codec L3_kernels)
set_target_properties(ssb_q21_fused_v10 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

add_executable(ssb_q31_fused_v10 v10/q31_fused_v10.cu)
target_include_directories(ssb_q31_fused_v10 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v7
)
target_link_libraries(ssb_q31_fused_v10 L3_codec L3_kernels)
set_target_properties(ssb_q31_fused_v10 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

add_executable(ssb_q41_fused_v10 v10/q41_fused_v10.cu)
target_include_directories(ssb_q41_fused_v10 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v7
)
target_link_libraries(ssb_q41_fused_v10 L3_codec L3_kernels)
set_target_properties(ssb_q41_fused_v10 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

add_executable(ssb_q11_fused_v10 v10/q11_fused_v10.cu)
target_include_directories(ssb_q11_fused_v10 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ssb_q11_fused_v10 L3_codec L3_kernels)
set_target_properties(ssb_q11_fused_v10 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

add_executable(compression_benchmark v10/compression_benchmark.cu)
target_include_directories(compression_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(compression_benchmark L3_codec L3_kernels)
set_target_properties(compression_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)
