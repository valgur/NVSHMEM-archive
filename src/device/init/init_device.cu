/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdio.h>
#include <algorithm>
#include <cuda_runtime.h>

#include "common/nvshmem_version.h"
#include "internal/common/nvshmem_internal.h"
#include "internal/util.h"
#include "device/pt-to-pt/proxy_device.cuh"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "common/nvshmem_common_ibgda.h"

__constant__ nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;
#endif

__constant__ nvshmemi_device_state_t nvshmemi_device_state_d;
const nvshmemi_version_t nvshmemi_device_lib_version = {
    NVSHMEM_INTERLIB_MAJOR_VERSION, NVSHMEM_INTERLIB_MINOR_VERSION, NVSHMEM_INTERLIB_PATCH_VERSION};
__constant__ nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_INTERLIB_MAJOR_VERSION, NVSHMEM_INTERLIB_MINOR_VERSION, NVSHMEM_INTERLIB_PATCH_VERSION};

#ifdef __CUDA_ARCH__
#ifdef __cplusplus
extern "C" {
#endif
__device__ void nvshmem_global_exit(int status);
#ifdef __cplusplus
}
#endif

__device__ void nvshmem_global_exit(int status) {
    if (nvshmemi_device_state_d.proxy > NVSHMEMI_PROXY_NONE) {
        nvshmemi_proxy_global_exit(status);
    } else {
        /* TODO: Add device side printing macros */
        printf(
            "Device side proxy was called, but is not supported under your configuration. "
            "Please unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY, or set it to false.\n");
        assert(0);
    }
}
#endif

void nvshmemi_check_state_and_init_d() {
    int status;
    int ret;

    if (!nvshmemi_is_nvshmem_bootstrapped)
        NVSHMEMI_ERROR_EXIT("nvshmem API called before nvshmem_init \n");
    if (!nvshmemi_is_nvshmem_initialized) {
        /* The fact that we can pass NVSHMEM_THREAD_SERIALIZED
         * here is an implementation detail. It should be fixed
         * if/when NVSHMEM_THREAD_* becomes significant. */
        status = nvshmemx_host_init(NVSHMEM_THREAD_SERIALIZED, &ret, 0, NULL,
                                    nvshmemi_device_lib_version, NULL);
        if (status) {
            NVSHMEMI_ERROR_EXIT("nvshmem initialization failed, exiting \n");
        }
        nvshmemx_host_finalize(NULL, NULL);
    }
}

void nvshmemi_get_mem_handle(void **dev_state_ptr, void **transport_dev_state_ptr) {
    int status = 0;
    status = cudaGetSymbolAddress(dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to access device state. %d\n", status);
        *dev_state_ptr = NULL;
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    status = cudaGetSymbolAddress(transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to access ibgda device state. %d\n", status);
        *transport_dev_state_ptr = NULL;
    }
#endif
}

int nvshmemi_init_thread(int requested_thread_support, int *provided_thread_support,
                         unsigned int bootstrap_flags, nvshmemx_init_attr_t *bootstrap_attr,
                         nvshmemi_version_t nvshmem_app_version) {
    int status = 0;

    if (nvshmemi_is_version_compatible(nvshmem_app_version, nvshmemi_device_lib_version) != 0) {
        printf(
            "NVSHMEM version used in application does not match with NVSHMEM device library "
            "version\n");
        return 1;
    }
    status =
        nvshmemx_host_init(requested_thread_support, provided_thread_support, bootstrap_flags,
                           bootstrap_attr, nvshmemi_device_lib_version, &nvshmemi_get_mem_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem_internal_init_thread failed \n");

out:
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif
void nvshmemi_finalize() {
    int status;
    void *dev_state_ptr, *transport_dev_state_ptr = NULL;

    status = cudaGetSymbolAddress(&dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
        nvshmemx_host_finalize(NULL, NULL);
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    status = cudaGetSymbolAddress(&transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
        nvshmemx_host_finalize(NULL, NULL);
    }
#endif
    nvshmemx_host_finalize(dev_state_ptr, transport_dev_state_ptr);
}
#ifdef __cplusplus
}
#endif
