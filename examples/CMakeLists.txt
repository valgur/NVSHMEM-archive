cmake_minimum_required(VERSION 3.18)
PROJECT(NVSHMEMExamples VERSION 1.0.0
	LANGUAGES CUDA CXX)

get_directory_property(SubBuild PARENT_DIRECTORY)

if(NOT SubBuild)
	find_package(NVSHMEM REQUIRED)
endif()

add_library(nvshmem_example_helper INTERFACE)
target_link_libraries(nvshmem_example_helper INTERFACE CUDA::cudart CUDA::cuda_driver
                                                 nvshmem)
target_compile_features(nvshmem_example_helper INTERFACE cxx_std_11 cuda_std_11)

if(NVSHMEM_SHMEM_SUPPORT)
  target_link_libraries(nvshmem_example_helper INTERFACE shmem)
  target_compile_definitions(nvshmem_example_helper INTERFACE NVSHMEM_SHMEM_SUPPORT)
endif()

if(NVSHMEM_MPI_SUPPORT)
   target_link_libraries(nvshmem_example_helper INTERFACE MPI::MPI_CXX)
   target_compile_definitions(nvshmem_example_helper INTERFACE NVSHMEM_MPI_SUPPORT)
endif()

 if (NVSHMEM_UCX_SUPPORT)
   target_link_libraries(nvshmem_example_helper INTERFACE ucx::ucs ucx::ucp)
   target_compile_definitions(nvshmem_example_helper INTERFACE NVSHMEM_UCX_SUPPORT)
endif()

if (NVSHMEM_IBDEVX_SUPPORT OR NVSHMEM_IBGDA_SUPPORT)
  target_link_libraries(nvshmem_example_helper INTERFACE ${MLX5_lib})
endif()

if (NVSHMEM_LIBFABRIC_SUPPORT)
  target_link_libraries(nvshmem_example_helper INTERFACE ${FABRIC_lib})
endif()

if(NOT NVSHMEM_NVTX)
  target_compile_definitions(nvshmem_example_helper INTERFACE NVTX_DISABLE)
endif()

# check: do we need this here ?
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(EXAMPLES
  collective-launch.cu
  on-stream.cu
  thread-group.cu
  put-block.cu
  dev-guide-ring.cu
  )

if(NVSHMEM_MPI_SUPPORT)
  LIST(APPEND EXAMPLES
    mpi-based-init.cu
    dev-guide-ring-mpi.cu
  )
endif()

foreach(example ${EXAMPLES})
  get_filename_component(NAME_ ${example} NAME_WE)
  add_executable(${NAME_} ${example})
  if(NVSHMEM_TEST_STATIC_LIB)
    target_link_libraries(${NAME_} nvshmem_host nvshmem_device nvshmem_example_helper)
  else()
    target_link_libraries(${NAME_} nvshmem nvshmem_example_helper)
  endif()
endforeach()
