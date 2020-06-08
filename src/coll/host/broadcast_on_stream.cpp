/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"

// need to use cpu shared mem to figure out offsets because CPU issues copy ops
int nvshmemxi_broadcast_cpu_ipc_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                   size_t nelems, int PE_root, int PE_start,
                                                   int logPE_stride, int PE_size, long *pSync,
                                                   cudaStream_t stream) {
    int status = 0;
    int offset;
    char *base = NULL;
    char *ipc_src = NULL;
    int msg_len = nelems * type_size;
    int stride = 1 << logPE_stride;
    int *wait_ptr;
    int wait_val = 0;
    int wait_flag = CU_STREAM_WAIT_VALUE_EQ;
    int *write_ptr;
    int write_val = 0;
    int write_flag = CU_STREAM_WRITE_VALUE_DEFAULT;
    volatile int *bcast_sync_arr = nvshm_cpu_coll_info.cpu_bcast_int_sync_arr;
    volatile int *bcast_data_arr = nvshm_cpu_coll_info.cpu_bcast_int_data_arr;
    volatile int *gpu_bcast_sync_arr = nvshm_cpu_coll_info.gpu_bcast_int_sync_arr;

    // calculate offset at root

    if (PE_root == nvshmem_state->mype) {
        // as a root, broadcast offset and release CPU but root stream
        // still waits for non-root streams to complete copying
        ipc_src = (char *)source;
        base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
        offset = ipc_src - base;

        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
                // do nothing
            }
        }

        bcast_data_arr[PE_root] = offset;

        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            bcast_sync_arr[ii] = 1;
        }

        // wait for data exposed to be read by non-roots
        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
                // do nothing
            }
        }

        // wait for all non-root streams to finish previous ops if not already
        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            wait_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            wait_val = 0;
            CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
        }

        // notify non-root streams
        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            write_val = 1;
            write_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        }

        // notify non-roots
        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            wait_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            wait_val = 0;
            CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
        }
    } else {
        // as a non-root, obtain offset first
        while (1 != bcast_sync_arr[nvshmem_state->mype]) {
            // do nothing
        }
        offset = bcast_data_arr[PE_root];
        // notify root that offset read is complete
        bcast_sync_arr[nvshmem_state->mype] = 0;
        // stream waits for root stream to notify arrival
        wait_ptr = (int *)&(gpu_bcast_sync_arr[nvshmem_state->mype]);
        wait_val = 1;
        CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
        // finish transfer dest<-source of msg_length
        ipc_src = (char *)nvshmem_state->peer_heap_base[PE_root] + offset;
        CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)dest, (CUdeviceptr)ipc_src, msg_len, stream));
        write_val = 0;
        write_ptr = (int *)&(gpu_bcast_sync_arr[nvshmem_state->mype]);
        CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
    }

fn_out:
    return status;
}

int nvshmemxi_broadcast_cpu_p2p_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                   size_t nelems, int PE_root, int PE_start,
                                                   int logPE_stride, int PE_size, long *pSync,
                                                   cudaStream_t stream) {
    int status = 0;
    if (nvshmem_state->mype != PE_root) {
        nvshmemx_getmem_on_stream(dest, source, nelems * type_size, PE_root, stream);
    }
    nvshmemx_barrier_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
fn_out:
    return status;
}

int nvshmemxi_broadcast_cpu_p2p_all_push_on_stream(void *dest, const void *source, int type_size,
                                                   size_t nelems, int PE_root, int PE_start,
                                                   int logPE_stride, int PE_size, long *pSync,
                                                   cudaStream_t stream) {
    int status = 0;
    int stride = 1 << logPE_stride;
    if (PE_root == nvshmem_state->mype) {
        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            nvshmemx_putmem_on_stream(dest, source, nelems * type_size, ii, stream);
        }
    }
    nvshmemx_barrier_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
fn_out:
    return status;
}

void nvshmemxi_broadcast_on_stream(void *dest, const void *source, int type_size, size_t nelems,
                                   int PE_root, int PE_start, int logPE_stride, int PE_size,
                                   long *pSync, cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    if (!nvshm_enable_p2p_cpu_coll) {
        nvshmemxi_broadcast_cpu_ipc_all_pull_on_stream(dest, source, type_size, nelems, PE_root,
                                                       PE_start, logPE_stride, PE_size, pSync,
                                                       stream);
    } else {
        if (!nvshm_use_p2p_cpu_push) {
            nvshmemxi_broadcast_cpu_p2p_all_pull_on_stream(dest, source, type_size, nelems, PE_root,
                                                           PE_start, logPE_stride, PE_size, pSync,
                                                           stream);
        } else {
            nvshmemxi_broadcast_cpu_p2p_all_push_on_stream(dest, source, type_size, nelems, PE_root,
                                                           PE_start, logPE_stride, PE_size, pSync,
                                                           stream);
        }
    }
}

#define DEFN_NVSHMEM_CPU_BCAST_ON_STREAM(BITS)                                                 \
    void nvshmemx_broadcast##BITS##_on_stream(void *dest, const void *source, size_t nelems,   \
                                              int PE_root, int PE_start, int logPE_stride,     \
                                              int PE_size, long *pSync, cudaStream_t stream) { \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                        \
        if (nvshm_use_tg_for_stream_coll) {                                                    \
            call_broadcast##BITS##_on_stream_kern(dest, source, nelems, PE_root, PE_start,     \
                                                  logPE_stride, PE_size, pSync, stream);       \
        } else {                                                                               \
            nvshmemxi_broadcast_on_stream(dest, source, (BITS / 8), nelems, PE_root, PE_start, \
                                          logPE_stride, PE_size, pSync, stream);               \
        }                                                                                      \
    }

DEFN_NVSHMEM_CPU_BCAST_ON_STREAM(32);
DEFN_NVSHMEM_CPU_BCAST_ON_STREAM(64);
