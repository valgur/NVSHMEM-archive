/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <stdio.h>
#include "nvshmemx_error.h"
#include "util.h"
#include <algorithm>


__constant__ nvshmemi_device_state_t nvshmemi_device_state_d;
bool nvshmemi_is_device_state_set = false;
#ifdef __CUDA_ARCH__
__device__ void nvshmem_global_exit(int status) {
    if (nvshmemi_device_state_d.proxy > NVSHMEMI_PROXY_NONE) {
        nvshmemi_proxy_global_exit(status);
    } else {
        /* TODO: Add device side printing macros */
        printf("Device side proxy was called, but is not supported under your configuration. "
               "Please unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY, or set it to false.\n");
        assert(0);
    }
}
#endif

int nvshmemi_set_device_state(nvshmemi_device_state_t *nvshmemi_device_state) {
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(nvshmemi_device_state_d,
										  (void *)nvshmemi_device_state,
										  sizeof(nvshmemi_device_state_t), 0, cudaMemcpyHostToDevice));
	nvshmemi_is_device_state_set = 1;
    return 0;
}

void nvshmemi_check_state_and_init() {
	if (!nvshmemi_is_device_state_set) {
		if (!nvshmemi_is_nvshmem_bootstrapped) ERROR_EXIT("nvshmem API called before nvshmem_init \n");
		if (!nvshmemi_is_nvshmem_initialized) {
			if (nvshmemx_internal_common_init()) {
				ERROR_EXIT("nvshmem initialization failed, exiting \n");
			}
		}
		nvshmemi_device_state_t *nvshmemi_device_state;
		nvshmemx_get_device_state(&nvshmemi_device_state);
		nvshmemi_set_device_state(nvshmemi_device_state);
	}
}

int nvshmemi_init_thread(int requested_thread_support,
						 int *provided_thread_support,
						 unsigned int bootstrap_flags,
						 nvshmemx_init_attr_t *bootstrap_attr) {
	int status = 0;
	nvshmemi_device_state_t *nvshmemi_device_state;
	nvshmemi_check_state_and_init_fn_ptr = &nvshmemi_check_state_and_init;
    nvshmemi_init_counter++;
	status = nvshmemx_internal_init_thread(requested_thread_support, provided_thread_support,
										   bootstrap_flags, bootstrap_attr);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_internal_init_thread failed \n");
	if (nvshmemi_is_nvshmem_initialized) {
		nvshmemx_get_device_state(&nvshmemi_device_state);
		nvshmemi_set_device_state(nvshmemi_device_state);
	}
out:
	return status;
}
