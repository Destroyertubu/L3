# L3 Examples

# Example 1: Partition Strategies
add_executable(example_partition_strategies
    cpp/01_partition_strategies.cpp
)

target_link_libraries(example_partition_strategies
    PRIVATE
        l3_compression
)

target_include_directories(example_partition_strategies
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Install examples
install(TARGETS example_partition_strategies
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples
)
