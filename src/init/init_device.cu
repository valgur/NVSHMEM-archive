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

extern int (*nvshmemi_transport_init_op[NVSHMEM_TRANSPORT_COUNT])(nvshmem_transport_t *transport);

void *heap_base_array_dptr = NULL;
void *heap_base_actual_array_dptr = NULL;
int nvshmemi_job_connectivity;

nvshmemi_device_state_t nvshmemi_device_state;
__constant__ nvshmemi_device_state_t nvshmemi_device_state_d;

int nvshmemi_proxy_level(nvshmemi_state_t *state) {
    bool need_proxy = false;
    int proxy_level = NVSHMEMI_PROXY_MINIMAL;

    if (nvshmemi_job_connectivity >= NVSHMEMI_JOB_GPU_LDST) {
        need_proxy = (state->transports[NVSHMEM_TRANSPORT_ID_IBRC] &&
                      nvshmemi_transport_init_op[NVSHMEM_TRANSPORT_ID_IBRC] &&
                      state->transports[NVSHMEM_TRANSPORT_ID_IBRC]->is_successfully_initialized);
        need_proxy |= (state->transports[NVSHMEM_TRANSPORT_ID_UCX] &&
                       nvshmemi_transport_init_op[NVSHMEM_TRANSPORT_ID_UCX] &&
                       state->transports[NVSHMEM_TRANSPORT_ID_UCX]->is_successfully_initialized);
    }

    if (need_proxy == true) {
        proxy_level = NVSHMEMI_PROXY_FULL;
    } else if (nvshmemi_options.DISABLE_LOCAL_ONLY_PROXY) {
        proxy_level = NVSHMEMI_PROXY_NONE;
    }

    return proxy_level;
}

int nvshmemi_set_device_state() {
    CUDA_RUNTIME_CHECK(cudaMemcpyToSymbol(nvshmemi_device_state_d, (void *)&nvshmemi_device_state,
        sizeof(nvshmemi_device_state_t), 0, cudaMemcpyHostToDevice));
    return 0;
}


int set_job_connectivity (nvshmemi_state_t *state) {
    int status;
    int *job_connectivity_all; 
    bool proxy_ops_are_ordered = true;

    // detrmine job level connectivity among GPUs
    nvshmemi_job_connectivity = NVSHMEMI_JOB_GPU_LDST_ATOMICS;
    for (int i = 0; i < state->npes; i++) {
        int peer_connectivity = NVSHMEMI_JOB_GPU_PROXY;
        void *enforce_cst = NULL;
        // for each PE, pick the best connectivity of any transport
        for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
            if (state->transports[j]) {
                if (state->transports[j]->cap[i] & NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS) {
                    peer_connectivity = (int)NVSHMEMI_JOB_GPU_LDST_ATOMICS;
                } else if (state->transports[j]->cap[i] &
                           (NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST | NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD)) {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_LDST);
                } else {
                    peer_connectivity = std::min(peer_connectivity, (int)NVSHMEMI_JOB_GPU_PROXY);
                    enforce_cst = (void *)state->transports[j]->host_ops.enforce_cst_at_target;
                }
            }
        }
        if ((peer_connectivity == NVSHMEMI_JOB_GPU_PROXY) && (enforce_cst)) {
            peer_connectivity = NVSHMEMI_JOB_GPU_PROXY_CST;
        }
        // for the job, pick the weakest connecitivity to any remote PEs
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, peer_connectivity);
    }

    //agree on maximumg distance for job_connectivity among all PEs
    job_connectivity_all = (int *)malloc(sizeof(int) * state->npes);
    NULL_ERROR_JMP(job_connectivity_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "memory allocation for job_connectivity_all failed \n");

    status = state->boot_handle.allgather((void *)&nvshmemi_job_connectivity, (void *)job_connectivity_all, 
                   sizeof(int), &state->boot_handle);
    if (status != 0) {
        free(job_connectivity_all);
        ERROR_PRINT("allgather of job_connectivity failed \n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    for (int i = 0; i < state->npes; i++) {
        nvshmemi_job_connectivity = std::max(nvshmemi_job_connectivity, job_connectivity_all[i]);
    }
    free(job_connectivity_all);

    nvshmemi_device_state.job_connectivity = nvshmemi_job_connectivity;

    // check if all proxy ops are ordered
    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (state->transports[i] && (state->transports[i]->host_ops.fence != NULL))
            proxy_ops_are_ordered = false;
    }
    nvshmemi_device_state.proxy_ops_are_ordered = proxy_ops_are_ordered;

out:
    return status;
}

int nvshmemi_init_device_state(nvshmemi_state_t *state) {
    int status = CUDA_SUCCESS;
    int warp_size = 0;

    status = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, state->cudevice );
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "querying warp size failed \n");
    if ( NVSHMEMI_WARP_SIZE != warp_size ) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                     "device warp size (%d) does not match assumed warp size (%d)\n",
                     warp_size, NVSHMEMI_WARP_SIZE);

    }

    status = cuMemAlloc((CUdeviceptr *)&heap_base_array_dptr, (state->npes) * sizeof(void *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                 "device peer heap base allocation failed \n");

    status =
        cuMemAlloc((CUdeviceptr *)&heap_base_actual_array_dptr, (state->npes) * sizeof(void *));
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                 "device peer heap base actual allocation failed \n");

    status = set_job_connectivity(state);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "set_job_connectivity failed \n");

    status =
        cuMemcpyHtoDAsync((CUdeviceptr)heap_base_array_dptr, (const void *)state->peer_heap_base,
                          sizeof(void *) * state->npes, state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "peer heap base initialization failed \n");

    status = cuMemcpyHtoDAsync((CUdeviceptr)heap_base_actual_array_dptr,
                               (const void *)state->peer_heap_base_actual,
                               sizeof(void *) * state->npes, state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "peer heap base actual initialization failed \n");

    status = cuStreamSynchronize(state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 " stream synchronize failed\n");

    nvshmemi_device_state.proxy = nvshmemi_proxy_level(state);

    if (nvshmemi_options.ASSERT_ATOMICS_SYNC) nvshmemi_device_state.atomics_sync = 1;
    else nvshmemi_device_state.atomics_sync = 0;

    nvshmemi_device_state.peer_heap_base =  (void **)heap_base_array_dptr;

    INFO(NVSHMEM_INIT,
         "[%d] status %d cudaErrorInvalidValue %d cudaErrorInvalidSymbol %d "
         "cudaErrorInvalidMemcpyDirection %d cudaErrorNoKernelImageForDevice %d",
         state->mype, status, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
         cudaErrorInvalidMemcpyDirection, cudaErrorNoKernelImageForDevice);

    nvshmemi_device_state.peer_heap_base_actual = (void **)heap_base_actual_array_dptr;
    nvshmemi_device_state.heap_base = state->heap_base;
    nvshmemi_device_state.heap_size = state->heap_size;
    nvshmemi_device_state.mype = state->mype;
    nvshmemi_device_state.npes = state->npes;
    nvshmemi_device_state.node_mype = state->mype_node;
    nvshmemi_device_state.node_npes = state->npes_node;
    nvshmemi_device_state.barrier_dissem_kval = nvshmemi_options.BARRIER_DISSEM_KVAL;
    nvshmemi_device_state.barrier_tg_dissem_kval = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;

    status = cuStreamSynchronize(state->my_stream);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, " stream synchronize failed\n");

    unsigned long long *test_wait_any_start_idx_ptr;
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&test_wait_any_start_idx_ptr, sizeof(unsigned long long)));
    CUDA_RUNTIME_CHECK(cudaMemset((void *)test_wait_any_start_idx_ptr, 0, sizeof(unsigned long long)));

    nvshmemi_device_state.test_wait_any_start_idx_ptr = test_wait_any_start_idx_ptr;
    nvshmemi_set_device_state();

out:
    if (status) {
        if (heap_base_array_dptr) free(heap_base_array_dptr);
        if (heap_base_actual_array_dptr) free(heap_base_actual_array_dptr);
        if (test_wait_any_start_idx_ptr) cudaFree(test_wait_any_start_idx_ptr);
    }
    return status;
}

int nvshmemx_cumodule_init(CUmodule module) {
    int status = 0;
    CUdeviceptr dptr;
    size_t size;
    status = cuModuleGetGlobal(&dptr, &size, module, "nvshmemi_device_state_d");
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuModuleGetGlobal failed\n");
    status = cudaMemcpyFromSymbol((void *)dptr, nvshmemi_device_state_d, size, 0,
                                            cudaMemcpyDeviceToDevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMemcpyFromSymbol failed\n");
    status = cudaDeviceSynchronize();
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cudaMemcpyFromSymbol failed\n");
out:
    return status;
}

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
