/*
 * * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "util.h"
#include "gpu_coll.h"
#include "nvshmem_internal.h"

__device__ volatile int *gpu_bcast_int_sync_arr_d;
__device__ volatile int *gpu_bcast_int_data_arr_d;
__device__ volatile char *gpu_own_intm_addr_d;
__device__ volatile char *gpu_own_intm_rdxn_addr_d;
__device__ volatile long *gpu_ipsync_d;
__device__ volatile int4 *gpu_ipwrk_d;
__device__ long *gpu_icounter_d;
__device__ long *gpu_icounter_barrier_d;
__device__ gpu_coll_env_params_t gpu_coll_env_params_var_d;


__device__ int reduce_recexch_step1_sendto_d;
__device__ int *reduce_recexch_step1_recvfrom_d;
__device__ int reduce_recexch_step1_nrecvs_d;
__device__ int **reduce_recexch_step2_nbrs_d;
__device__ int reduce_recexch_step2_nphases_d;
__device__ int reduce_recexch_p_of_k_d;
__device__ int reduce_recexch_reduce_recexch_digit_d;
__device__ int *digit_d;

extern "C" int init_shm_kernel_shm_ptr(gpu_coll_info_t *nvshm_gpu_coll_info) {
    int status = 0;

    int *step1_recvfrom = NULL, **step2_nbrs = NULL;
    int *digit = NULL;
    int k, max_phases;
    status = cudaMemcpyToSymbol(gpu_own_intm_addr_d, &(nvshm_gpu_coll_info->own_intm_addr),
                                sizeof(volatile char *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status =
        cudaMemcpyToSymbol(gpu_own_intm_rdxn_addr_d, &(nvshm_gpu_coll_info->own_intm_rdxn_addr),
                           sizeof(volatile char *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status =
        cudaMemcpyToSymbol(gpu_ipsync_d, &(nvshm_gpu_coll_info->ipsync), sizeof(volatile long *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbol(gpu_icounter_d, &(nvshm_gpu_coll_info->icounter), sizeof(long *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbol(gpu_icounter_barrier_d, &(nvshm_gpu_coll_info->icounter_barrier),
                                sizeof(long *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbol(gpu_ipwrk_d, &(nvshm_gpu_coll_info->ipwrk), sizeof(int4 *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbol(gpu_coll_env_params_var_d, &gpu_coll_env_params_var,
                                sizeof(gpu_coll_env_params_t));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    /* Allocate memory for performing reduce recursive exchange algorithm */
    k = gpu_coll_env_params_var.reduce_recexch_kval;
    assert(k > 1);

    status = cuMemAlloc((CUdeviceptr *) &step1_recvfrom, sizeof(int) * (k - 1));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc failed\n");

    status = cudaMemcpyToSymbol(reduce_recexch_step1_recvfrom_d, &step1_recvfrom, sizeof(void *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    max_phases = log(nvshmem_state->npes) / log(k) + 1; /* The '+ 1' makes it a conservative calculation, max_pahses >= 1 */

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

    status = cudaMemcpyToSymbol(reduce_recexch_step2_nbrs_d, &step2_nbrs, sizeof(void**));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");
    status = cudaMemcpyToSymbol(digit_d, &digit, sizeof(int *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "memcopy to symbol failed \n");

    nvshmemi_recexchalgo_get_neighbors(nvshmem_state->mype, nvshmem_state->npes);

    CUDA_CHECK(cuStreamSynchronize(0));

    /*

    nvshmemi_kern_fxn_ptrs_init<<<1, 1>>>();

    status = cudaMemcpyToSymbol(gpu_rdxn_fptr_arr_d, gpu_rdxn_fptr_arr,
                                sizeof(gpu_rdxn_fxn_ptr_t) * gpu_rd_null * gpu_rd_dt_null);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "memcopy to symbol failed \n");

    status = cudaMemcpyToSymbol(gpu_cumemcpy_fptr_arr_d, gpu_cumemcpy_fptr_arr,
                                sizeof(gpu_cumemcpy_fxn_ptr_t) * gpu_rd_dt_null);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "memcopy to symbol failed \n");
    */

    goto fn_out;
out:
fn_out:
    return status;
}
