/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "gpu_coll.h"
#include "cpu_coll.h"
#include "nvshmem_internal.h"

gpu_coll_info_t nvshm_gpu_coll_info;
gpu_coll_env_params_t gpu_coll_env_params_var;
int nvshm_gpu_coll_initialized = 0;

int nvshmemi_coll_common_gpu_read_env() {
    int status = 0;

    gpu_coll_env_params_var.reduce_recexch_kval = nvshmemi_options.REDUCE_RECEXCH_KVAL;

    if (gpu_coll_env_params_var.reduce_recexch_kval > nvshmem_state->npes)
        gpu_coll_env_params_var.reduce_recexch_kval = max(2, nvshmem_state->npes);

    return status;
}

int nvshmemi_coll_common_gpu_init_memory() {
    int status = NVSHMEMI_COLL_GPU_STATUS_SUCCESS;
    char *tmp = NULL;
    int page_size = getpagesize();
    int i;

    size_t alloc_size = GPU_SCRATCH_SIZE + (GPU_IPSYNC_SIZE * nvshmem_state->npes) +
                        GPU_RDXN_SCRATCH_SIZE + GPU_ICOUNTER_BARRIER_SIZE + GPU_ICOUNTER_SIZE +
                        GPU_IPWRK_SIZE;
    char *base_ptr = (char *)nvshmemi_malloc(alloc_size);
    if (!base_ptr) {
        fprintf(stderr, "nvshmemi_malloc failed \n");
        goto fn_out;
    }

    nvshm_gpu_coll_info.own_intm_addr = (volatile char *)base_ptr;
    base_ptr += GPU_SCRATCH_SIZE;

    nvshm_gpu_coll_info.ipsync = (volatile long int *)base_ptr;
    base_ptr += (GPU_IPSYNC_SIZE * nvshmem_state->npes);

    tmp = (char *)calloc((GPU_IPSYNC_SIZE * nvshmem_state->npes), sizeof(char));
    CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)nvshm_gpu_coll_info.ipsync, (const void *)tmp,
                            (GPU_IPSYNC_SIZE * nvshmem_state->npes)));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, fn_fail, "error in cuMemcpyHtoD\n");
    CUDA_CHECK(cuStreamSynchronize(0));
    free(tmp);

    nvshm_gpu_coll_info.own_intm_rdxn_addr = (volatile char *)base_ptr;
    base_ptr += GPU_RDXN_SCRATCH_SIZE;
    gpu_coll_env_params_var.gpu_intm_rdxn_size = GPU_RDXN_SCRATCH_SIZE;

    nvshm_gpu_coll_info.icounter = (volatile long *)base_ptr;
    base_ptr += GPU_ICOUNTER_SIZE;

    tmp = (char *)malloc(GPU_ICOUNTER_SIZE);
    for (i = 0; i < SYNC_SIZE; i++) {
        *((long *)tmp + i) = 1;
    }
    CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)nvshm_gpu_coll_info.icounter, (const void *)tmp,
                            GPU_ICOUNTER_SIZE));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, fn_fail, "error in cuMemcpyHtoD\n");
    CUDA_CHECK(cuStreamSynchronize(0));
    free(tmp);

    nvshm_gpu_coll_info.icounter_barrier = (volatile long *)base_ptr;
    base_ptr += GPU_ICOUNTER_BARRIER_SIZE;

    tmp = (char *)malloc(GPU_ICOUNTER_BARRIER_SIZE);
    for (i = 0; i < SYNC_SIZE; i++) {
        *((long *)tmp + i) = 1;
    }
    CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)nvshm_gpu_coll_info.icounter_barrier, (const void *)tmp,
                            GPU_ICOUNTER_BARRIER_SIZE));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, fn_fail, "error in cuMemcpyHtoD\n");
    CUDA_CHECK(cuStreamSynchronize(0));
    free(tmp);

    // What happens if the compiler doesn't support int4?
    nvshm_gpu_coll_info.ipwrk = (volatile int4 *)base_ptr;
    if (!nvshm_gpu_coll_info.ipwrk) {
        fprintf(stderr, "nvshmemi_malloc failed \n");
        goto fn_out;
    }

    tmp = (char *)calloc(GPU_IPWRK_SIZE, sizeof(char));
    CUDA_CHECK(
        cuMemcpyHtoD((CUdeviceptr)nvshm_gpu_coll_info.ipwrk, (const void *)tmp, GPU_IPWRK_SIZE));
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, fn_fail, "error in cuMemcpyHtoD\n");
    CUDA_CHECK(cuStreamSynchronize(0));
    free(tmp);

    init_shm_kernel_shm_ptr(&nvshm_gpu_coll_info);

    /*add barrier so that ipsync arrays aren't touched by other procs*/
    status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
    assert(status == 0);

    goto fn_out;
fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_coll_common_gpu_init() {
    int status = NVSHMEMI_COLL_GPU_STATUS_SUCCESS;

    // Read env var list to select appropriate designs and

    /* FIXME: Should this become an env var? */
    gpu_coll_env_params_var.gpu_intm_rdxn_size = -1;
    gpu_coll_env_params_var.reduce_recexch_kval = 2;

    status = nvshmemi_coll_common_gpu_read_env();
    if (status) NVSHMEMI_COLL_GPU_ERR_POP();

    status = nvshmemi_coll_common_gpu_init_memory();
    if (status) NVSHMEMI_COLL_GPU_ERR_POP();

    nvshm_gpu_coll_initialized = 1;

    goto fn_out;
fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_coll_common_gpu_finalize() {
    int status = NVSHMEMI_COLL_GPU_STATUS_SUCCESS;

    if (0 == nvshm_gpu_coll_initialized) return status;

    nvshmemi_free((void *)nvshm_gpu_coll_info.own_intm_addr);
    nvshm_gpu_coll_initialized = 0;

    goto fn_out;
fn_out:
    return status;
}
