/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "cpu_coll.h"

int nvshmemi_collect_cpu_all_bcast(void *dest, const void *source, int type_size, size_t nelems,
                                   int PE_start, int logPE_stride, int PE_size, long *pSync) {
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

        nvshmemi_broadcast_cpu_ipc_all_pull(round_dest, source, type_size, nelems, round_root,
                                            PE_start, logPE_stride, PE_size, pSync);
        // status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
        // if (status) NVSHMEMI_COLL_CPU_ERR_POP();
    }

    offset = ((nvshmem_state->mype - PE_start) * type_size * nelems);
    round_dest = (void *)((char *)dest + offset);
    CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)round_dest, (CUdeviceptr)source, msg_len,
                                 nvshmem_state->my_stream));
    CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));

fn_out:
    return status;
}

int nvshmemi_collect_cpu_p2p_all_pull(void *dest, const void *source, int type_size, size_t nelems,
                                      int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int next_rank;
    int next_offset;
    int mype = nvshmem_state->mype;
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = (mype + (ii * stride)) % (stride * PE_size);
        next_offset = nelems * type_size * ((next_rank - PE_start) / stride);
        nvshmem_getmem((void *)((char *)dest + next_offset), source, nelems * type_size, next_rank);
    }
    nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);
fn_out:
    return status;
}

int nvshmemi_collect_cpu_p2p_all_push(void *dest, const void *source, int type_size, size_t nelems,
                                      int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int next_rank;
    int next_offset;
    int mype = nvshmem_state->mype;
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = (mype + (ii * stride)) % (stride * PE_size);
        next_offset = nelems * type_size * ((mype - PE_start) / stride);
        nvshmem_putmem((void *)((char *)dest + next_offset), source, nelems * type_size, next_rank);
    }
    nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);
fn_out:
    return status;
}

void nvshmemi_collect(void *dest, const void *source, int type_size, size_t nelems, int PE_start,
                      int logPE_stride, int PE_size, long *pSync) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    if (!nvshm_enable_p2p_cpu_coll) {
        nvshmemi_collect_cpu_all_bcast(dest, source, type_size, nelems, PE_start, logPE_stride,
                                       PE_size, pSync);
    } else {
        if (!nvshm_use_p2p_cpu_push) {
            nvshmemi_collect_cpu_p2p_all_pull(dest, source, type_size, nelems, PE_start,
                                              logPE_stride, PE_size, pSync);
        } else {
            nvshmemi_collect_cpu_p2p_all_push(dest, source, type_size, nelems, PE_start,
                                              logPE_stride, PE_size, pSync);
        }
    }
}

#define DEFN_NVSHMEM_CPU_COLLECT(BITS)                                                          \
    void nvshmem_collect##BITS(void *dest, const void *source, size_t nelems, int PE_start,     \
                               int logPE_stride, int PE_size, long *pSync) {                    \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                         \
        if (nvshm_use_tg_for_cpu_coll) {                                                        \
            call_collect##BITS##_on_stream_kern(dest, source, nelems, PE_start, logPE_stride,   \
                                                PE_size, pSync, nvshmem_state->my_stream);      \
            CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));                          \
        } else {                                                                                \
            nvshmemi_collect(dest, source, (BITS / 8), nelems, PE_start, logPE_stride, PE_size, \
                             pSync);                                                            \
        }                                                                                       \
    }

DEFN_NVSHMEM_CPU_COLLECT(32);
DEFN_NVSHMEM_CPU_COLLECT(64);
