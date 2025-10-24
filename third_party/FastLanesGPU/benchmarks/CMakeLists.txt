add_executable(benchmark_bitpack benchmark_bitpack.cu)
target_link_libraries(benchmark_bitpack PUBLIC tilebased)
target_include_directories(benchmark_bitpack PUBLIC
    ${CMAKE_SOURCE_DIR}/tile_based/src/include
    ${CMAKE_SOURCE_DIR}/tile_based/src
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(benchmark_delta benchmark_delta.cu)
target_link_libraries(benchmark_delta PUBLIC tilebased)
target_include_directories(benchmark_delta PUBLIC
    ${CMAKE_SOURCE_DIR}/tile_based/src/include
    ${CMAKE_SOURCE_DIR}/tile_based/src
    ${CMAKE_SOURCE_DIR}/include
)
