/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvshmem.h"
#include "error_codes_internal.h"
#include "debug.h"
#include "cudawrap.h"

#include <dlfcn.h>

#define DECLARE_CUDA_PFN(symbol, version) PFN_##symbol##_v##version pfn_##symbol = nullptr

DECLARE_CUDA_PFN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN(cuCtxSynchronize, 2000);
DECLARE_CUDA_PFN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN(cuPointerSetAttribute, 6000);
DECLARE_CUDA_PFN(cuModuleGetGlobal, 3020);
DECLARE_CUDA_PFN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN(cuDevicePrimaryCtxRetain, 7000);
DECLARE_CUDA_PFN(cuCtxGetCurrent, 4000);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 11000
DECLARE_CUDA_PFN(cuMemCreate, 10020);
DECLARE_CUDA_PFN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN(cuMemMap, 10020);
DECLARE_CUDA_PFN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemRelease, 10020);
DECLARE_CUDA_PFN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN(cuMemUnmap, 10020);
#endif

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN(cuInit, 2000);
DECLARE_CUDA_PFN(cuGetProcAddress, 11030);

static enum { cudaUninitialized, cudaInitializing, cudaInitialized, cudaError } cudaState = cudaUninitialized;

static void *cudaLib;
static int cudaDriverVersion;

static int cudaPfnFuncLoader(void) {
  CUresult res;

#define LOAD_SYM(symbol, version, sym_suffix, ignore) do {           \
    bool not_found = false;                                             \
    if(pfn_cuGetProcAddress) {                                          \
        res = pfn_cuGetProcAddress(#symbol, (void **) (&pfn_##symbol), version, 0); \
        if (res != 0) not_found = true;                                 \
    }                                                                   \
    else {                                                              \
      pfn_##symbol = (PFN_##symbol##_v##version) dlsym(cudaLib, #symbol#sym_suffix);            \
      if (pfn_##symbol == NULL) not_found = true;                       \
    }                                                                   \
    if (not_found) {                                                    \
      if (!ignore) {                                                    \
        WARN("Retrieve %s version %d failed", #symbol#sym_suffix, cudaDriverVersion); \
        return NVSHMEMI_SYSTEM_ERROR; }                                       \
    } } while(0)

  LOAD_SYM(cuCtxGetDevice, 2000, , 0);
  LOAD_SYM(cuCtxSynchronize, 2000, , 0);
  LOAD_SYM(cuDeviceGet, 2000, , 0);
  LOAD_SYM(cuDeviceGetAttribute, 2000, , 0);
  LOAD_SYM(cuPointerSetAttribute, 6000, , 0);
  LOAD_SYM(cuModuleGetGlobal, 3020, _v2, 0);
  LOAD_SYM(cuGetErrorString, 6000, , 0);
  LOAD_SYM(cuGetErrorName, 6000, , 0);
  LOAD_SYM(cuCtxSetCurrent, 4000, , 0);
  LOAD_SYM(cuDevicePrimaryCtxRetain, 7000, , 0);
  LOAD_SYM(cuCtxGetCurrent, 4000, , 0);
#if CUDA_VERSION >= 11070
  LOAD_SYM(cuMemGetHandleForAddressRange, 11070, , 1); // DMA-BUF support
#endif
#if CUDA_VERSION >= 11000
  LOAD_SYM(cuMemCreate, 10020, , 1);
  LOAD_SYM(cuMemMap, 10020, , 1);
  LOAD_SYM(cuMemAddressReserve, 10020, , 1);
  LOAD_SYM(cuMemGetAllocationGranularity, 10020, , 1);
  LOAD_SYM(cuMemImportFromShareableHandle, 10020, , 1);
  LOAD_SYM(cuMemExportToShareableHandle, 10020, , 1);
  LOAD_SYM(cuMemRelease, 10020, , 1);
  LOAD_SYM(cuMemSetAccess, 10020, , 1);
  LOAD_SYM(cuMemUnmap, 10020, , 1);
#endif
  return NVSHMEMI_SUCCESS;
}

int nvshmemi_cuda_library_init(void) {
  cudaError_t cuda_err;

  if (cudaState == cudaInitialized)
    return NVSHMEMI_SUCCESS;
  if (cudaState == cudaError)
    return NVSHMEMI_SYSTEM_ERROR;

  /*
   * Load CUDA driver library
   */
  char path[1024];
  char *nvshmemCudaPath = getenv("NVSHMEM_CUDA_PATH");
  if (nvshmemCudaPath == NULL)
    snprintf(path, 1024, "%s", "libcuda.so");
  else
    snprintf(path, 1024, "%s/%s", nvshmemCudaPath, "libcuda.so");

  cudaLib = dlopen(path, RTLD_LAZY);
  if (cudaLib == NULL) {
    WARN("Failed to find CUDA library in %s (NVSHMEM_CUDA_PATH=%s)", nvshmemCudaPath, nvshmemCudaPath);
    goto error;
  }

  /*
   * Load initial CUDA functions
   */

  pfn_cuInit = (PFN_cuInit_v2000) dlsym(cudaLib, "cuInit");
  if (pfn_cuInit == NULL) {
    WARN("Failed to load CUDA missing symbol cuInit");
    goto error;
  }

  cuda_err = cudaDriverGetVersion(&cudaDriverVersion);
  if (cuda_err != 0) {
    WARN("cudaDriverGetVersion failed with %d", cuda_err);
    goto error;
  }
  INFO(NVSHMEM_INIT, "cudaDriverVersion %d", cudaDriverVersion);

  pfn_cuGetProcAddress = (PFN_cuGetProcAddress_v11030) dlsym(cudaLib, "cuGetProcAddress");

  /*
   * Required to initialize the CUDA Driver.
   * Multiple calls of cuInit() will return immediately
   * without making any relevant change
   */
  pfn_cuInit(0);

  if (cudaPfnFuncLoader()) {
    WARN("CUDA some PFN functions not found in the library");
    goto error;
  }

  cudaState = cudaInitialized;
  return NVSHMEMI_SUCCESS;

error:
  cudaState = cudaError;
  return NVSHMEMI_SYSTEM_ERROR;
}
