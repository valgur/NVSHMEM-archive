/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"

int nvshmemxi_scatter_cpu_ipc_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                 size_t nelems, int PE_root, int PE_start,
                                                 int logPE_stride, int PE_size, long *pSync,
                                                 cudaStream_t stream) {
    int status = 0;
    int offset;
    int offset_for_me;
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

        offset = ((nvshmem_state->mype - PE_start) * type_size * nelems);
        ipc_src = (char *)source + offset;

        // wait for data exposed to be read by non-roots
        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
                // do nothing
            }
        }

        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            wait_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            wait_val = 0;
            CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
        }

        for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
            if (PE_root == ii) continue;
            write_val = 1;
            write_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        }

        CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)dest, (CUdeviceptr)ipc_src, msg_len, stream));

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
        offset_for_me = offset + (((nvshmem_state->mype - PE_start) / stride) * msg_len);
        bcast_sync_arr[nvshmem_state->mype] = 0;
        ipc_src = (char *)nvshmem_state->peer_heap_base[PE_root] + offset_for_me;
        wait_ptr = (int *)&(gpu_bcast_sync_arr[nvshmem_state->mype]);
        wait_val = 1;
        CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
        CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)dest, (CUdeviceptr)ipc_src, msg_len, stream));
        write_val = 0;
        write_ptr = (int *)&(gpu_bcast_sync_arr[nvshmem_state->mype]);
        CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
    }

fn_out:
    return status;
}

int nvshmemxi_alltoall_cpu_ipc_all_scatter_on_stream(void *dest, const void *source, int type_size,
                                                     size_t nelems, int PE_start, int logPE_stride,
                                                     int PE_size, long *pSync,
                                                     cudaStream_t stream) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int offset;
    int round_root;
    void *round_dest;

    // pes take turn to be a root
    for (int ii = PE_start; ii < (stride * PE_size); ii = (ii + stride)) {
        round_root = ii;
        offset = (round_root * type_size * nelems);
        round_dest = (void *)((char *)dest + offset);

        nvshmemxi_scatter_cpu_ipc_all_pull_on_stream(round_dest, source, type_size, nelems,
                                                     round_root, PE_start, logPE_stride, PE_size,
                                                     pSync, stream);
    }

fn_out:
    return status;
}

int nvshmemxi_alltoall_cpu_p2p_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                  size_t nelems, int PE_start, int logPE_stride,
                                                  int PE_size, long *pSync, cudaStream_t stream) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int next_rank;
    int src_offset;
    int dst_offset;
    int mype = nvshmem_state->mype;
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = (mype + (ii * stride)) % (stride * PE_size);
        src_offset = nelems * type_size * ((mype - PE_start) / stride);
        dst_offset = nelems * type_size * ((next_rank - PE_start) / stride);
        nvshmemx_getmem_nbi_on_stream((void *)((char *)dest + dst_offset),
                                      (void *)((char *)source + src_offset), nelems * type_size,
                                      next_rank, stream);
    }
    nvshmemx_barrier_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
fn_out:
    return status;
}

int nvshmemxi_alltoall_cpu_p2p_all_push_on_stream(void *dest, const void *source, int type_size,
                                                  size_t nelems, int PE_start, int logPE_stride,
                                                  int PE_size, long *pSync, cudaStream_t stream) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int next_rank;
    int src_offset;
    int dst_offset;
    int mype = nvshmem_state->mype;
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = (mype + (ii * stride)) % (stride * PE_size);
        src_offset = nelems * type_size * ((next_rank - PE_start) / stride);
        dst_offset = nelems * type_size * ((mype - PE_start) / stride);
        nvshmemx_putmem_nbi_on_stream((void *)((char *)dest + dst_offset),
                                      (void *)((char *)source + src_offset), nelems * type_size,
                                      next_rank, stream);
    }
    nvshmemx_barrier_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
fn_out:
    return status;
}

void nvshmemxi_alltoall_on_stream(void *dest, const void *source, int type_size, size_t nelems,
                                  int PE_start, int logPE_stride, int PE_size, long *pSync,
                                  cudaStream_t stream) {
    if (!nvshm_enable_p2p_cpu_coll) {
        nvshmemxi_alltoall_cpu_ipc_all_scatter_on_stream(dest, source, type_size, nelems, PE_start,
                                                         logPE_stride, PE_size, pSync, stream);
    } else {
        if (!nvshm_use_p2p_cpu_push) {
            nvshmemxi_alltoall_cpu_p2p_all_pull_on_stream(dest, source, type_size, nelems, PE_start,
                                                          logPE_stride, PE_size, pSync, stream);
        } else {
            nvshmemxi_alltoall_cpu_p2p_all_push_on_stream(dest, source, type_size, nelems, PE_start,
                                                          logPE_stride, PE_size, pSync, stream);
        }
    }
}

#define DEFN_NVSHMEM_CPU_ALLTOALL_ON_STREAM(BITS)                                                  \
    void nvshmemx_alltoall##BITS##_on_stream(void *dest, const void *source, size_t nelems,        \
                                             int PE_start, int logPE_stride, int PE_size,          \
                                             long *pSync, cudaStream_t stream) {                   \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        if (nvshm_use_tg_for_stream_coll) {                                                        \
            call_alltoall##BITS##_on_stream_kern(dest, source, nelems, PE_start, logPE_stride,     \
                                                 PE_size, pSync, stream);                          \
        } else {                                                                                   \
            nvshmemxi_alltoall_on_stream(dest, source, (BITS / 8), nelems, PE_start, logPE_stride, \
                                         PE_size, pSync, stream);                                  \
        }                                                                                          \
    }

DEFN_NVSHMEM_CPU_ALLTOALL_ON_STREAM(32);
DEFN_NVSHMEM_CPU_ALLTOALL_ON_STREAM(64);
