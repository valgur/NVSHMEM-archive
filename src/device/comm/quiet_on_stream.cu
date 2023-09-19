/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "device/nvshmem_defines.h"
#include "internal/util.h"

#ifdef __cplusplus
extern "C" {
#endif
__device__ void nvshmem_quiet();
#ifdef __cplusplus
}
#endif

__global__ void nvshmemi_proxy_quiet_entrypoint() { nvshmem_quiet(); }

void nvshmemi_call_proxy_quiet_entrypoint(cudaStream_t cstrm) {
    int status =
        cudaLaunchKernel((const void *)nvshmemi_proxy_quiet_entrypoint, 1, 1, NULL, 0, cstrm);
    if (status) {
        NVSHMEMI_ERROR_PRINT("cudaLaunchKernel() failed in nvshmem_quiet_on_stream \n");
    }
}
