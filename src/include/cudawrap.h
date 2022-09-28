/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NVSHMEM_CUDAWRAP_H
#define NVSHMEM_CUDAWRAP_H

#include <cuda.h>

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#else
typedef CUresult (CUDAAPI *PFN_cuInit_v2000)(unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuGetProcAddress_v11030)(const char *symbol, void **pfn, int driverVersion, cuuint64_t flags);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetAttribute_v2000)(int* pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuPointerSetAttribute_v6000)(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr);
typedef CUresult (CUDAAPI *PFN_cuGetErrorString_v6000)(CUresult error, const char** pStr);
typedef CUresult (CUDAAPI *PFN_cuGetErrorName_v6000)(CUresult error, const char** pStr);
typedef CUresult (CUDAAPI *PFN_cuDeviceGet_v2000)(CUdevice* device, int ordinal);
typedef CUresult (CUDAAPI *PFN_cuCtxSetCurrent_v4000)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxGetDevice_v2000)(CUdevice * device);
typedef CUresult (CUDAAPI *PFN_cuCtxGetCurrent_v4000)(CUcontext* pctx);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRetain_v7000)(CUcontext* pctx, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize_v2000)();
typedef CUresult (CUDAAPI *PFN_cuModuleGetGlobal_v3020)(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
#if CUDA_VERSION >= 11000
typedef CUresult (CUDAAPI *PFN_cuMemCreate_v10020)(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemGetAllocationGranularity_v10020)(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option);
typedef CUresult (CUDAAPI *PFN_cuMemAddressReserve_v10020)(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemExportToShareableHandle_v10020)(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemImportFromShareableHandle_v10020)(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType);
typedef CUresult (CUDAAPI *PFN_cuMemMap_v10020)(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemRelease_v10020)(CUmemGenericAllocationHandle handle);
typedef CUresult (CUDAAPI *PFN_cuMemSetAccess_v10020)(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
typedef CUresult (CUDAAPI *PFN_cuMemUnmap_v10020)(CUdeviceptr ptr, size_t size);
typedef CUresult (CUDAAPI *PFN_cuMemGetAccess_v10020)(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr);
#endif
#endif

#define CUPFN(symbol) pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECKNORETURN(cmd) do {	              \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				          \
      (void) pfn_cuGetErrorString(err, &errStr);  \
      WARN("Cuda failure '%s'", errStr);		  \
    }							                  \
    assert(err == CUDA_SUCCESS);                  \
} while(false)

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure '%s'", errStr);		      \
      return NVSHMEMI_UNHANDLED_CUDA_ERROR;			      \
    }							      \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure '%s'", errStr);		      \
      res = NVSHMEMI_UNHANDLED_CUDA_ERROR;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    CUresult err = pfn_##cmd;						\
    if( err != CUDA_SUCCESS ) {						\
      const char *errStr;						\
      (void) pfn_cuGetErrorString(err, &errStr);			\
      INFO(NCCL_ALL,"%s:%d Cuda failure '%s'", __FILE__, __LINE__, errStr);	\
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    CUresult err = pfn_##cmd;						\
    if (err != CUDA_SUCCESS) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = NVSHMEMI_UNHANDLED_CUDA_ERROR;				\
      return args;							\
    }									\
} while(0)

#define DECLARE_CUDA_PFN_EXTERN(symbol, version) extern PFN_##symbol##_v##version pfn_##symbol

DECLARE_CUDA_PFN_EXTERN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN_EXTERN(cuPointerSetAttribute, 6000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN_EXTERN(cuCtxSynchronize, 2000);
DECLARE_CUDA_PFN_EXTERN(cuModuleGetGlobal, 3020);
DECLARE_CUDA_PFN_EXTERN(cuDevicePrimaryCtxRetain, 7000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent, 4000);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 11000
DECLARE_CUDA_PFN_EXTERN(cuMemCreate, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemMap, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRelease, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemUnmap, 10020);
#endif

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN_EXTERN(cuInit, 2000);
DECLARE_CUDA_PFN_EXTERN(cuGetProcAddress, 11030);


int nvshmemi_cuda_library_init(void);

#endif
