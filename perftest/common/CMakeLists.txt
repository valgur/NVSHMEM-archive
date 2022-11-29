add_library(nvshmem_perftest_helper STATIC utils.cu)
target_include_directories(nvshmem_perftest_helper
                           PUBLIC ${CMAKE_SOURCE_DIR}/perftest/common/)
target_link_libraries(nvshmem_perftest_helper PUBLIC CUDA::cudart CUDA::cuda_driver
                                                 nvshmem)
target_compile_features(nvshmem_perftest_helper PUBLIC cxx_std_11 cuda_std_11)

if(NVSHMEM_SHMEM_SUPPORT)
  target_link_libraries(nvshmem_perftest_helper PUBLIC shmem)
  target_compile_definitions(nvshmem_perftest_helper PUBLIC NVSHMEM_SHMEM_SUPPORT)
endif()

if(NVSHMEM_MPI_SUPPORT)
   target_link_libraries(nvshmem_perftest_helper PUBLIC MPI::MPI_CXX)
   target_compile_definitions(nvshmem_perftest_helper PUBLIC NVSHMEM_MPI_SUPPORT)
endif()

 if (NVSHMEM_UCX_SUPPORT)
   target_link_libraries(nvshmem_perftest_helper PUBLIC ucx::ucs ucx::ucp)
   target_compile_definitions(nvshmem_perftest_helper PUBLIC NVSHMEM_UCX_SUPPORT)
endif()

if (NVSHMEM_IBDEVX_SUPPORT OR NVSHMEM_IBGDA_SUPPORT)
  target_link_libraries(nvshmem_perftest_helper PUBLIC ${MLX5_lib})
endif()

if (NVSHMEM_LIBFABRIC_SUPPORT)
  target_link_libraries(nvshmem_perftest_helper PUBLIC ${FABRIC_lib})
endif()

target_compile_options(nvshmem_perftest_helper
PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
$<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
)

if(NOT NVSHMEM_NVTX)
  target_compile_definitions(nvshmem_perftest_helper PUBLIC NVTX_DISABLE)
endif()
