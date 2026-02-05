if (NOT BUILD_SSB_ONLY)
  add_executable(compress_index compress_index.cpp)
  target_link_libraries(compress_index
    gpu_integers_compression
    CLI11
  )

  add_executable(perf_decode perf_decode.cpp)
  target_link_libraries(perf_decode
    gpu_integers_compression
    CLI11
  )
endif ()

if(BUILD_CUDA)
if (NOT BUILD_SSB_ONLY)
  CUDA_ADD_EXECUTABLE(cuda_compress_index cuda_compress_index.cu)
  target_link_libraries(cuda_compress_index
    gpu_integers_compression
    CLI11
  )
  CUDA_ADD_EXECUTABLE(cuda_perf_decode cuda_perf_decode.cu)
  target_link_libraries(cuda_perf_decode
    gpu_integers_compression
    CLI11
  )
endif ()

CUDA_ADD_EXECUTABLE(ssb_gpu_bp
  ssb/ssb_main.cu
  ssb/q1.cu
  ssb/q2.cu
  ssb/q3.cu
  ssb/q4.cu
)
target_link_libraries(ssb_gpu_bp
  gpu_integers_compression
)
target_include_directories(ssb_gpu_bp BEFORE PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
endif()
