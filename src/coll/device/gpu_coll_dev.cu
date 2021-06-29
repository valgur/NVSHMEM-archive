/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "util.h"
#include "gpu_coll.h"
#include "nvshmem_internal.h"

extern "C" int init_shm_kernel_shm_ptr() {
    int status = 0;

    int *step1_recvfrom = NULL, **step2_nbrs = NULL;
    int *digit = NULL;
    int k, max_phases;

    nvshmemi_device_state.gpu_coll_env_params_var = gpu_coll_env_params_var;

    /* Allocate memory for performing reduce recursive exchange algorithm */
    k = gpu_coll_env_params_var.reduce_recexch_kval;
    assert(k > 1);

    status = cuMemAlloc((CUdeviceptr *) &step1_recvfrom, sizeof(int) * (k - 1));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");

    nvshmemi_device_state.reduce_recexch_step1_recvfrom = step1_recvfrom;

    max_phases = log(nvshmemi_state->npes) / log(k) + 1; /* The '+ 1' makes it a conservative calculation, max_pahses >= 1 */

    status = cuMemAlloc((CUdeviceptr *) &step2_nbrs, sizeof(int *) * max_phases);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");
    status = cuMemAlloc((CUdeviceptr *) &digit, sizeof(int) * max_phases);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");

    for (int i = 0; i < max_phases; i++) {
        void *dev_ptr;
        status = cudaMalloc(&dev_ptr, sizeof(int) * (k - 1));
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");
        status = cuMemcpyHtoD((CUdeviceptr)((int **)step2_nbrs + i), &dev_ptr, sizeof(int *));
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpyHtoD failed\n");
    }

    nvshmemi_device_state.reduce_recexch_step2_nbrs = step2_nbrs;
    nvshmemi_device_state.digit = digit;
    nvshmemi_recexchalgo_get_neighbors(nvshmemi_state->mype, nvshmemi_state->npes);
    nvshmemi_set_device_state();
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    goto fn_out;
out:
fn_out:
    return status;
}
