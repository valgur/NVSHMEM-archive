# this requires cmake 3.19
include(CheckCompilerFlag)

check_compiler_flag(CUDA -t4 NVCC_THREADS)

# Start bootstrap definitions
add_library(
  nvshmem_bootstrap_pmi SHARED
)
set(ACTIVE_BOOTSTRAPS nvshmem_bootstrap_pmi)
target_sources(nvshmem_bootstrap_pmi PRIVATE bootstrap/bootstrap_pmi.cpp pmi/simple-pmi/simple_pmi.cpp pmi/simple-pmi/simple_pmiutil.cpp)
target_include_directories(nvshmem_bootstrap_pmi
                           PRIVATE include
                                   bootstrap
                                   pmi/simple-pmi
)

add_library(
  nvshmem_bootstrap_pmi2 SHARED
)
set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmi2)
target_sources(nvshmem_bootstrap_pmi2 PRIVATE bootstrap/bootstrap_pmi.cpp pmi/pmi-2/pmi2_api.c pmi/pmi-2/pmi2_util.c)
target_compile_definitions(nvshmem_bootstrap_pmi2 PRIVATE NVSHMEM_BUILD_PMI2)
target_link_options(nvshmem_bootstrap_pmi2 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_bootstrap.sym")
target_include_directories(nvshmem_bootstrap_pmi2
                           PRIVATE include
                                   bootstrap
                                   pmi/pmi-2
)

if(NVSHMEM_PMIX_SUPPORT)
  add_library(
    nvshmem_bootstrap_pmix SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmix)
  target_sources(nvshmem_bootstrap_pmix PRIVATE bootstrap/bootstrap_pmix.c)
  find_library(PMIX_lib NAMES pmix HINTS "${PMIX_HOME}/lib")
  target_link_libraries(nvshmem_bootstrap_pmix PRIVATE ${PMIX_lib})
  target_include_directories(nvshmem_bootstrap_pmix
                             PRIVATE ${PMIX_HOME}/include
                             include
                             bootstrap
  )
else()
set(NVSHMEM_DEFAULT_PMIX OFF)
endif()

if(NVSHMEM_MPI_SUPPORT)
  add_library(
    nvshmem_bootstrap_mpi SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_mpi)
  target_sources(nvshmem_bootstrap_mpi PRIVATE bootstrap/bootstrap_mpi.c)
  target_link_libraries(nvshmem_bootstrap_mpi PRIVATE MPI::MPI_CXX)
  target_include_directories(nvshmem_bootstrap_mpi
                             PRIVATE include
                             bootstrap
  )
endif()

if(NVSHMEM_SHMEM_SUPPORT)
  add_library(
    nvshmem_bootstrap_shmem SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_shmem)
  target_sources(nvshmem_bootstrap_shmem PRIVATE bootstrap/bootstrap_shmem.c)
  target_link_libraries(nvshmem_bootstrap_shmem PRIVATE shmem)
  target_include_directories(nvshmem_bootstrap_shmem
                             PRIVATE include
                             bootstrap
  )
endif()

set_target_properties(${ACTIVE_BOOTSTRAPS} PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")

# End bootstrap definitions

# Start nvshmem lib definitions
add_library(
  nvshmem STATIC
)

add_library(
  nvshmem_device STATIC
)

add_library(
  nvshmem_host SHARED
)

## Start generic variable configuration
set_target_properties(nvshmem nvshmem_host nvshmem_device
                      PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CXX_STANDARD_REQUIRED ON
                      CUDA_STANDARD_REQUIRED ON
                      CXX_STANDARD 11
                      CUDA_STANDARD 11
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
		      VERSION ${PROJECT_VERSION}
		      SOVERSION ${PROJECT_VERSION_MAJOR}
)

macro(nvshmem_library_set_base_config LIBNAME)
  target_include_directories(
    ${LIBNAME}
    PRIVATE include
            util
            bootstrap
            comm
            coll/host
            coll/device
            coll
            comm/transports/common
            topo
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
              $<INSTALL_INTERFACE:include>
  )

  target_compile_definitions(${LIBNAME}
    PRIVATE $<$<CONFIG:Debug>:_NVSHMEM_DEBUG;ENABLE_TRACE;NVSHMEM_IBGDA_DEBUG>
    $<IF:$<BOOL:${NVSHMEM_NVTX}>,,NVTX_DISABLE>
    $<IF:$<BOOL:${NVSHMEM_MPI_SUPPORT}>,NVSHMEM_MPI_SUPPORT,>
    $<IF:$<BOOL:${NVSHMEM_SHMEM_SUPPORT}>,NVSHMEM_SHMEM_SUPPORT,>
    $<IF:$<BOOL:${NVSHMEM_UCX_SUPPORT}>,NVSHMEM_UCX_SUPPORT,>
    $<IF:$<BOOL:${NVSHMEM_LIBFABRIC_SUPPORT}>,NVSHMEM_LIBFABRIC_SUPPORT,>
    $<IF:$<BOOL:${NVSHMEM_IBRC_SUPPORT}>,NVSHMEM_IBRC_SUPPORT,>
    $<IF:$<BOOL:${NVSHMEM_IBDEVX_SUPPORT}>,NVSHMEM_IBDEVX_SUPPORT,>
    $<IF:$<BOOL:${NVSHMEM_IBGDA_SUPPORT}>,NVSHMEM_IBGDA_SUPPORT,>
    $<IF:$<BOOL:${NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY}>,NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY,>
    $<IF:$<BOOL:${NVSHMEM_USE_GDRCOPY}>,NVSHMEM_USE_GDRCOPY,>
    $<IF:$<BOOL:${NVSHMEM_USE_NCCL}>,NVSHMEM_USE_NCCL,>
    $<IF:$<BOOL:${NVSHMEM_DISABLE_COLL_POLL}>,NVSHMEM_DISABLE_COLL_POLL,>
    $<IF:$<BOOL:${NVSHMEM_GPU_COLL_USE_LDST}>,NVSHMEM_GPU_COLL_USE_LDST,>
    $<IF:$<BOOL:${NVSHMEM_TIMEOUT_DEVICE_POLLING}>,NVSHMEM_TIMEOUT_DEVICE_POLLING,>
    $<IF:$<BOOL:${NVSHMEM_TRACE}>,ENABLE_TRACE,>
    $<IF:$<BOOL:${NVSHMEM_DEFAULT_PMIX}>,NVSHMEM_DEFAULT_PMIX,>
    $<IF:$<AND:$<NOT:$<BOOL:${NVSHMEM_DEFAULT_PMIX}>>,$<BOOL:${NVSHMEM_DEFAULT_PMI2}>>,NVSHMEM_DEFAULT_PMI2,>
    $<IF:$<BOOL:${NVSHMEM_DEFAULT_UCX}>,NVSHMEM_DEFAULT_UCX,>
    $<IF:$<BOOL:${NVSHMEM_COMPLEX_SUPPORT}>,NVSHMEM_COMPLEX_SUPPORT,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","x86_64">,NVSHMEM_X86_64,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","ppc64le">,__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS;NVSHMEM_PPC64LE,>
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
  )

  target_compile_options(${LIBNAME}
    INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVSHMEM_VERBOSE}>>:-Xptxas -v>
    PRIVATE $<IF:$<CONFIG:Debug>,-O0;-g;,-O3>
    $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    $<IF:$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>,-O0;-g;-G;,-O3>
    $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},"x86_64">,-msse,>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  )

  if(NVSHMEM_DEVEL)
    target_compile_options(
      ${LIBNAME}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror
              all-warnings>
              $<$<COMPILE_LANGUAGE:CXX>:-Werror>
              -Wall
              -Wextra
              -Wno-unused-function
              -Wno-unused-parameter
    )
  endif()

  target_link_libraries(${LIBNAME} PRIVATE CUDA::cudart CUDA::cuda_driver nvidia-ml)
endmacro()

set(NVSHMEM_DEVICE_SOURCES
    coll/device/alltoall.cu
    coll/device/barrier.cu
    coll/device/broadcast.cu
    coll/device/fcollect.cu
    coll/device/gpu_coll.cu
    coll/device/gpu_coll_dev.cu
    coll/device/recexchalgo.cu
    coll/device/rdxn_thread.cu
    coll/device/rdxn_warp.cu
    coll/device/rdxn_block.cu
    comm/device/proxy_device.cu
    comm/device/transfer_device.cu
    init/init_device.cu
    init/query_device.cu
    launch/collective_launch_device.cu
    team/team_device.cu
    team/team_internal_device.cu
)

set(NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT
    coll/device/kernels/alltoall.cu
    coll/device/kernels/barrier.cu
    coll/device/kernels/broadcast.cu
    coll/device/kernels/fcollect.cu
    coll/device/kernels/rdxn.cu
    comm/host/cuda_interface_sync.cu
    comm/host/proxy/rma.cu
    comm/host/quiet_on_stream.cu
)

set(NVSHMEM_HOST_SOURCES
    bootstrap/bootstrap.cpp
    bootstrap/bootstrap_loader.cpp
    coll/host/cpu_coll.cpp
    coll/host/alltoall.cpp
    coll/host/alltoall_on_stream.cpp
    coll/host/barrier.cpp
    coll/host/barrier_on_stream.cpp
    coll/host/broadcast.cpp
    coll/host/broadcast_on_stream.cpp
    coll/host/fcollect.cpp
    coll/host/fcollect_on_stream.cpp
    coll/host/rdxn.cpp
    coll/host/rdxn_on_stream.cpp
    comm/host/putget.cpp
    comm/host/fence.cpp
    comm/host/quiet.cpp
    comm/host/sync.cpp
    comm/host/amo.cpp
    comm/proxy/proxy.cpp
    comm/transport.cpp
    comm/transports/p2p/p2p.cpp
    init/cudawrap.cpp
    init/init.cu
    init/init_nvtx.cpp
    init/query_host.cpp
    launch/collective_launch.cpp
    mem/mem.cpp
    team/team.cu
    team/team_internal.cu
    topo/topo.cpp
    util/cs.cpp
    util/debug.cpp
    util/env_vars.cpp
    util/util.cpp
    util/sockets.cpp
)

nvshmem_library_set_base_config(nvshmem)
nvshmem_library_set_base_config(nvshmem_host)
nvshmem_library_set_base_config(nvshmem_device)
## End generic variable configuration

## Start dependencies
if(NVSHMEM_USE_GDRCOPY)
  target_link_libraries(nvshmem PRIVATE gdrcopy)
  target_link_libraries(nvshmem_host PRIVATE gdrcopy)
endif()

if(NVSHMEM_USE_NCCL)
  target_link_libraries(nvshmem PRIVATE nccl)
  target_link_libraries(nvshmem_host PRIVATE nccl)
endif()

if(NVSHMEM_USE_DLMALLOC)
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      mem/dlmalloc.cpp
  )
else()
set(NVSHMEM_HOST_SOURCES
    ${NVSHMEM_HOST_SOURCES}
    mem/custom_malloc.cpp
  )
endif()

if(NVSHMEM_IBRC_SUPPORT OR 
   NVSHMEM_IBDEVX_SUPPORT OR 
   NVSHMEM_IBGDA_SUPPORT OR 
   NVSHMEM_LIBFABRIC_SUPPORT OR 
   NVSHMEM_UCX_SUPPORT
)
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      comm/transports/common/transport_common.cpp
  )
  if(NVSHMEM_IBRC_SUPPORT OR 
     NVSHMEM_IBDEVX_SUPPORT OR 
     NVSHMEM_IBGDA_SUPPORT
  )
    set(NVSHMEM_HOST_SOURCES
        ${NVSHMEM_HOST_SOURCES}
        comm/transports/common/transport_ib_common.cpp
    )
  endif()
endif()

if(NVSHMEM_UCX_SUPPORT)
  target_link_libraries(nvshmem PRIVATE ucx::ucs ucx::ucp)
  target_link_libraries(nvshmem_host PRIVATE ucx::ucs ucx::ucp)

  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      comm/transports/ucx/ucx.cpp
  )
endif()

if(NVSHMEM_IBDEVX_SUPPORT OR NVSHMEM_IBGDA_SUPPORT)
  find_library(MLX5_lib NAMES mlx5)
  target_link_libraries(nvshmem PRIVATE ${MLX5_lib})
endif()

if(NVSHMEM_IBRC_SUPPORT)
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      comm/transports/ibrc/ibrc.cpp
  )
endif()

if(NVSHMEM_IBDEVX_SUPPORT)
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      comm/transports/ibdevx/ibdevx.cpp
  )
endif()

if(NVSHMEM_IBGDA_SUPPORT)
  set(NVSHMEM_DEVICE_SOURCES
      ${NVSHMEM_DEVICE_SOURCES}
      init/ibgda_init_device.cu
  )
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      init/ibgda_init.cu
      comm/transports/ibgda/ibgda.cpp
  )
endif()

if(NVSHMEM_LIBFABRIC_SUPPORT)
  find_library(FABRIC_lib NAMES fabric HINTS "${LIBFABRIC_HOME}/lib" "${LIBFABRIC_HOME}/lib64")
  target_link_libraries(nvshmem PRIVATE ${FABRIC_lib})
  target_link_libraries(nvshmem_host PRIVATE ${FABRIC_lib})
  target_include_directories(nvshmem PRIVATE "${LIBFABRIC_HOME}/include/")
  target_include_directories(nvshmem_host PRIVATE "${LIBFABRIC_HOME}/include/")
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      comm/transports/libfabric/libfabric.cpp
  )
endif()
## End dependecies

## Start final lib prep
target_link_libraries(nvshmem_host PRIVATE nvshmem_device)
target_link_options(nvshmem_host PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_host.sym")
set_source_files_properties(${NVSHMEM_HOST_SOURCES} ${NVSHMEM_DEVICE_SOURCES} PROPERTIES COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>)
target_sources(nvshmem PRIVATE ${NVSHMEM_HOST_SOURCES} ${NVSHMEM_DEVICE_SOURCES} ${NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT})
target_sources(nvshmem_host PRIVATE ${NVSHMEM_HOST_SOURCES})
target_sources(nvshmem_device PRIVATE ${NVSHMEM_DEVICE_SOURCES} ${NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT})

add_library(NVSHMEM::nvshmem ALIAS nvshmem)

## End final lib prep
# End nvshmem lib definitions

# Start nvshmem-info definitions.
add_executable(nvshmem-info)
target_sources(nvshmem-info PRIVATE util/nvshmem-info.cpp)
set_target_properties(nvshmem-info PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 11 RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
if(NVSHMEM_TEST_STATIC_LIB)
target_link_libraries(nvshmem-info CUDA::cudart CUDA::cuda_driver nvidia-ml nvshmem_host nvshmem_device)
else()
target_link_libraries(nvshmem-info CUDA::cudart CUDA::cuda_driver nvidia-ml nvshmem)
endif()
target_include_directories(nvshmem-info PRIVATE include)
# End nvshmem-info definitions.

# Start header configuration and export
configure_file(include/nvshmem_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/nvshmem_version.h @ONLY)

set(INCEXPORTS
    ${INCEXPORTS}
    nvshmem.h
    nvshmem_api.h
    nvshmem_coll_api.h
    nvshmem_common.cuh
    nvshmem_constants.h
    nvshmem_defines.h
    nvshmem_types.h
    nvshmem_version.h
    nvshmemi_constants.h
    nvshmemi_transfer.h
    nvshmemi_util.h
    nvshmemx.h
    nvshmemx_api.h
    nvshmemx_coll_api.h
    nvshmemx_defines.h
    nvshmemx_error.h
)

foreach(INC ${INCEXPORTS})
  configure_file(include/${INC} ${CMAKE_CURRENT_BINARY_DIR}/include/${INC} COPYONLY)
endforeach(INC ${INCEXPORTS})
# End header configuration and export

# start share configuration
set(SHAREEXPORTS
    bootstrap_loader.cpp
    bootstrap_mpi.c
    bootstrap_pmi.cpp
    bootstrap_pmix.c
    bootstrap_shmem.c
    bootstrap_util.h
    bootstrap.cpp
)

foreach(SHARE ${SHAREEXPORTS})
  configure_file(bootstrap/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/nvshmem/src/bootstrap-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${SHAREEXPORTS})
# end share configuration

# Start Installation definitions
install(
  TARGETS nvshmem nvshmem_host nvshmem_device nvshmem-info ${ACTIVE_BOOTSTRAPS}
  EXPORT NVSHMEMTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/share/nvshmem/src/bootstrap-plugins/
  DESTINATION share/nvshmem/src/bootstrap-plugins/
)
include(CMakePackageConfigHelpers)
# End Installation definitions

