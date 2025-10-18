# L3 Partitioner Module

# Source files
set(PARTITIONER_SOURCES
    fixed_size_partitioner.cpp
    variable_length_partitioner.cu
)

# Create object library
add_library(l3_partitioner OBJECT ${PARTITIONER_SOURCES})

target_include_directories(l3_partitioner
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA properties
set_target_properties(l3_partitioner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# CUDA architectures
set_property(TARGET l3_partitioner PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

# Link CUDA runtime
target_link_libraries(l3_partitioner
    PRIVATE
        CUDA::cudart
)
