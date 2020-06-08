/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"

int nvshmemxi_collect_cpu_all_bcast_on_stream(void *dest, const void *source, int type_size,
                                              size_t nelems, int PE_start, int logPE_stride,
                                              int PE_size, long *pSync, cudaStream_t stream) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int offset;
    int msg_len = nelems * type_size;
    int round_root;
    void *round_dest;

    // pes take turn to be a root
    for (int ii = PE_start; ii < (stride * PE_size); ii = (ii + stride)) {
        round_root = ii;
        offset = (round_root * type_size * nelems);
        round_dest = (void *)((char *)dest + offset);

        nvshmemxi_broadcast_cpu_ipc_all_pull_on_stream(round_dest, source, type_size, nelems,
                                                       round_root, PE_start, logPE_stride, PE_size,
                                                       pSync, stream);

        // not strictly necessary
        // status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
        // if (status) NVSHMEMI_COLL_CPU_ERR_POP();
    }
    offset = ((nvshmem_state->mype - PE_start) * type_size * nelems);
    round_dest = (char *)dest + offset;
    CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)round_dest, (CUdeviceptr)source, msg_len, stream));

fn_out:
    return status;
}

int nvshmemxi_collect_cpu_p2p_all_pull_on_stream(void *dest, const void *source, int type_size,
                                                 size_t nelems, int PE_start, int logPE_stride,
                                                 int PE_size, long *pSync, cudaStream_t stream) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int next_rank;
    int next_offset;
    int mype = nvshmem_state->mype;
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = (mype + (ii * stride)) % (stride * PE_size);
        next_offset = nelems * type_size * ((next_rank - PE_start) / stride);
        nvshmemx_getmem_on_stream((void *)((char *)dest + next_offset), source, nelems * type_size,
                                  next_rank, stream);
    }
    nvshmemx_barrier_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
fn_out:
    return status;
}

int nvshmemxi_collect_cpu_p2p_all_push_on_stream(void *dest, const void *source, int type_size,
                                                 size_t nelems, int PE_start, int logPE_stride,
                                                 int PE_size, long *pSync, cudaStream_t stream) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int next_rank;
    int next_offset;
    int mype = nvshmem_state->mype;
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = (mype + (ii * stride)) % (stride * PE_size);
        next_offset = nelems * type_size * ((mype - PE_start) / stride);
        nvshmemx_putmem_on_stream((void *)((char *)dest + next_offset), source, nelems * type_size,
                                  next_rank, stream);
    }
    nvshmemx_barrier_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
fn_out:
    return status;
}

void nvshmemxi_collect_on_stream(void *dest, const void *source, int type_size, size_t nelems,
                                 int PE_start, int logPE_stride, int PE_size, long *pSync,
                                 cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmemxi_collect_cpu_all_bcast_on_stream(dest, source, type_size, nelems, PE_start,
                                              logPE_stride, PE_size, pSync, stream);
    if (!nvshm_enable_p2p_cpu_coll) {
        nvshmemxi_collect_cpu_all_bcast_on_stream(dest, source, type_size, nelems, PE_start,
                                                  logPE_stride, PE_size, pSync, stream);
    } else {
        if (!nvshm_use_p2p_cpu_push) {
            nvshmemxi_collect_cpu_p2p_all_pull_on_stream(dest, source, type_size, nelems, PE_start,
                                                         logPE_stride, PE_size, pSync, stream);
        } else {
            nvshmemxi_collect_cpu_p2p_all_push_on_stream(dest, source, type_size, nelems, PE_start,
                                                         logPE_stride, PE_size, pSync, stream);
        }
    }
}

#define DEFN_NVSHMEM_CPU_COLLECT_ON_STREAM(BITS)                                                  \
    void nvshmemx_collect##BITS##_on_stream(void *dest, const void *source, size_t nelems,        \
                                            int PE_start, int logPE_stride, int PE_size,          \
                                            long *pSync, cudaStream_t stream) {                   \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                           \
        if (nvshm_use_tg_for_stream_coll) {                                                       \
            call_collect##BITS##_on_stream_kern(dest, source, nelems, PE_start, logPE_stride,     \
                                                PE_size, pSync, stream);                          \
        } else {                                                                                  \
            nvshmemxi_collect_on_stream(dest, source, (BITS / 8), nelems, PE_start, logPE_stride, \
                                        PE_size, pSync, stream);                                  \
        }                                                                                         \
    }

DEFN_NVSHMEM_CPU_COLLECT_ON_STREAM(32);
DEFN_NVSHMEM_CPU_COLLECT_ON_STREAM(64);
