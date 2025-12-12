# SSB Ultra - SSB Benchmark Tests for L3 Compression
# Implements all 13 SSB queries with three strategies

cmake_minimum_required(VERSION 3.18)

# Use parent project's CUDA configuration
enable_language(CUDA)

# Set include directories
set(SSB_ULTRA_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/kernels/decompression
)

# Helper function to create SSB query executable
function(add_ssb_query STRATEGY QUERY_NAME)
    set(TARGET_NAME ssb_${QUERY_NAME}_${STRATEGY})
    add_executable(${TARGET_NAME}
        ${STRATEGY}/${QUERY_NAME}.cu
    )
    target_include_directories(${TARGET_NAME} PRIVATE ${SSB_ULTRA_INCLUDES})
    target_link_libraries(${TARGET_NAME}
        L3_codec
        L3_kernels
        L3_codec
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
        OUTPUT_NAME "ssb_${QUERY_NAME}_${STRATEGY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# ============================================================================
# Decompress-First Strategy (baseline)
# Full decompression before query execution
# ============================================================================

# Q1 series
add_ssb_query(decompress_first q11)
add_ssb_query(decompress_first q12)
add_ssb_query(decompress_first q13)

# Q2 series
add_ssb_query(decompress_first q21)
add_ssb_query(decompress_first q22)
add_ssb_query(decompress_first q23)

# Q3 series
add_ssb_query(decompress_first q31)
add_ssb_query(decompress_first q32)
add_ssb_query(decompress_first q33)
add_ssb_query(decompress_first q34)

# Q4 series
add_ssb_query(decompress_first q41)
add_ssb_query(decompress_first q42)
add_ssb_query(decompress_first q43)

# ============================================================================
# Fused Query Strategy
# On-the-fly decompression with streaming processing
# ============================================================================

# Q1 series
add_ssb_query(fused_query q11)
add_ssb_query(fused_query q12)
add_ssb_query(fused_query q13)

# Q2 series
add_ssb_query(fused_query q21)
add_ssb_query(fused_query q22)
add_ssb_query(fused_query q23)

# Q3 series
add_ssb_query(fused_query q31)
add_ssb_query(fused_query q32)
add_ssb_query(fused_query q33)
add_ssb_query(fused_query q34)

# Q4 series
add_ssb_query(fused_query q41)
add_ssb_query(fused_query q42)
add_ssb_query(fused_query q43)

# ============================================================================
# Predicate Pushdown Strategy
# Use partition min/max to skip irrelevant partitions
# ============================================================================

# Q1 series
add_ssb_query(predicate_pushdown q11)
add_ssb_query(predicate_pushdown q12)
add_ssb_query(predicate_pushdown q13)

# Q2 series
add_ssb_query(predicate_pushdown q21)
add_ssb_query(predicate_pushdown q22)
add_ssb_query(predicate_pushdown q23)

# Q3 series
add_ssb_query(predicate_pushdown q31)
add_ssb_query(predicate_pushdown q32)
add_ssb_query(predicate_pushdown q33)
add_ssb_query(predicate_pushdown q34)

# Q4 series
add_ssb_query(predicate_pushdown q41)
add_ssb_query(predicate_pushdown q42)
add_ssb_query(predicate_pushdown q43)

# ============================================================================
# Random Access Strategy
# JOIN key full decompress + non-JOIN column random access
# Optimal for low-selectivity queries
# ============================================================================

# Q1 series
add_ssb_query(random_access q11)
add_ssb_query(random_access q12)
add_ssb_query(random_access q13)

# Q2 series
add_ssb_query(random_access q21)
add_ssb_query(random_access q22)
add_ssb_query(random_access q23)

# Q3 series
add_ssb_query(random_access q31)
add_ssb_query(random_access q32)
add_ssb_query(random_access q33)
add_ssb_query(random_access q34)

# Q4 series
add_ssb_query(random_access q41)
add_ssb_query(random_access q42)
add_ssb_query(random_access q43)

# ============================================================================
# Optimized Strategy
# Phase 1-5 optimizations: Two-Level Fast Hash, Shared Memory Cache, etc.
# ============================================================================

add_subdirectory(optimized)

# ============================================================================
# True Fused Decompression Strategy (NEW)
# Decompression inside query kernel - no intermediate global memory writes
# ============================================================================

add_subdirectory(fused_decompression)

# ============================================================================
# Crystal-opt Style Fused Query (NEW)
# FSL-GPU/Crystal-opt style implementation with L3 decompression
# ============================================================================

add_subdirectory(fused_query_new)

# ============================================================================
# Summary: Total 52+ executables (13 queries × 4 strategies + optimized)
# ============================================================================
message(STATUS "SSB Ultra: Building 52+ SSB query implementations")
message(STATUS "  - decompress_first: Q1.1-Q4.3 (13 queries)")
message(STATUS "  - fused_query: Q1.1-Q4.3 (13 queries)")
message(STATUS "  - predicate_pushdown: Q1.1-Q4.3 (13 queries)")
message(STATUS "  - random_access: Q1.1-Q4.3 (13 queries)")
message(STATUS "  - optimized: Two-Level Fast Hash + more optimizations")
