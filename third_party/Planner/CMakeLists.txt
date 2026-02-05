cmake_minimum_required(VERSION 3.18)

project(Planner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(ssb_planner
  src/ssb/ssb_main.cu
  src/ssb/q1.cu
  src/ssb/q2.cu
  src/ssb/q3.cu
  src/ssb/q4.cu
)

target_include_directories(ssb_planner PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(ssb_planner PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
