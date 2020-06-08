/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "cpu_coll.h"

int nvshmemi_scatter_cpu_ipc_all_pull(void *dest, const void *source, int type_size, size_t nelems,
                                      int PE_root, int PE_start, int logPE_stride, int PE_size,
                                      long *pSync) {
    int status = 0;
    int offset;
    int offset_for_me;
    char *base = NULL;
    char *ipc_src = NULL;
    int msg_len = nelems * type_size;
    int stride = 1 << logPE_stride;
    volatile int *bcast_sync_arr = nvshm_cpu_coll_info.cpu_bcast_int_sync_arr;
    volatile int *bcast_data_arr = nvshm_cpu_coll_info.cpu_bcast_int_data_arr;

    // calculate offset at root

    ipc_src = (char *)source;
    base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
    offset = ipc_src - base;

    if (PE_root == nvshmem_state->mype) {
        if (nvshm_cpu_coll_sync_reqd) {
            for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
                if (PE_root == ii) continue;
                while (0 != bcast_sync_arr[ii]) {
                    // do nothing
                }
            }
        }

        if (nvshm_cpu_coll_offset_reqd) {
            bcast_data_arr[PE_root] = offset;
        }

        if (nvshm_cpu_coll_sync_reqd) {
            for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
                if (PE_root == ii) continue;
                bcast_sync_arr[ii] = 1;
            }
        }

        offset = ((nvshmem_state->mype - PE_start) * type_size * nelems);
        ipc_src = (char *)((char *)source + offset);
        CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)dest, (CUdeviceptr)ipc_src, msg_len,
                                     nvshmem_state->my_stream));
        CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));

        if (nvshm_cpu_coll_sync_reqd) {
            // wait for data exposed to be read by non-roots
            for (int ii = PE_start; ii < (stride * PE_size); ii += stride) {
                if (PE_root == ii) continue;
                while (0 != bcast_sync_arr[ii]) {
                    // do nothing
                }
            }
        }

    } else {
        if (nvshm_cpu_coll_sync_reqd) {
            while (1 != bcast_sync_arr[nvshmem_state->mype]) {
                // do nothing
            }
        }

        if (nvshm_cpu_coll_offset_reqd) {
            offset = bcast_data_arr[PE_root];
        }
        offset_for_me = offset + (((nvshmem_state->mype - PE_start) / stride) * msg_len);

        // finish transfer dest<-source of msg_length
        ipc_src = (char *)nvshmem_state->peer_heap_base[PE_root] + offset_for_me;
        CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)dest, (CUdeviceptr)ipc_src, msg_len,
                                     nvshmem_state->my_stream));
        CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));

        if (nvshm_cpu_coll_sync_reqd) {
            bcast_sync_arr[nvshmem_state->mype] = 0;
        }
    }

fn_out:
    return status;
}

int nvshmemi_alltoall_cpu_ipc_all_scatter(void *dest, const void *source, int type_size,
                                          size_t nelems, int PE_start, int logPE_stride,
                                          int PE_size, long *pSync) {
    int status = 0;
    int stride = 1 << logPE_stride;
    int offset;
    int round_root;
    void *round_dest;

    // pes take turn to be a root
    for (int ii = PE_start; ii < (stride * PE_size); ii = (ii + stride)) {
        round_root = ii;
        offset = (round_root * type_size * nelems);
        round_dest = (char *)dest + offset;
        nvshmemi_scatter_cpu_ipc_all_pull(round_dest, source, type_size, nelems, round_root,
                                          PE_start, logPE_stride, PE_size, pSync);

        // status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
        // if (status) NVSHMEMI_COLL_CPU_ERR_POP();
    }

fn_out:
    return status;
}

int nvshmemi_alltoall_cpu_p2p_all_pull(void *dest, const void *source, int type_size, size_t nelems,
                                       int PE_start, int logPE_stride, int PE_size, long *pSync) {
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
        nvshmem_getmem((void *)((char *)dest + dst_offset), (void *)((char *)source + src_offset),
                       nelems * type_size, next_rank);
    }
    nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);
fn_out:
    return status;
}

int nvshmemi_alltoall_cpu_p2p_all_push(void *dest, const void *source, int type_size, size_t nelems,
                                       int PE_start, int logPE_stride, int PE_size, long *pSync) {
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
        nvshmem_putmem((void *)((char *)dest + dst_offset), (void *)((char *)source + src_offset),
                       nelems * type_size, next_rank);
    }
    nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);
fn_out:
    return status;
}

void nvshmemi_alltoall(void *dest, const void *source, int type_size, size_t nelems, int PE_start,
                       int logPE_stride, int PE_size, long *pSync) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    if (!nvshm_enable_p2p_cpu_coll) {
        nvshmemi_alltoall_cpu_ipc_all_scatter(dest, source, type_size, nelems, PE_start,
                                              logPE_stride, PE_size, pSync);
    } else {
        if (!nvshm_use_p2p_cpu_push) {
            nvshmemi_alltoall_cpu_p2p_all_pull(dest, source, type_size, nelems, PE_start,
                                               logPE_stride, PE_size, pSync);
        } else {
            nvshmemi_alltoall_cpu_p2p_all_push(dest, source, type_size, nelems, PE_start,
                                               logPE_stride, PE_size, pSync);
        }
    }
}

#define DEFN_NVSHMEM_CPU_ALLTOALL(BITS)                                                          \
    void nvshmem_alltoall##BITS(void *dest, const void *source, size_t nelems, int PE_start,     \
                                int logPE_stride, int PE_size, long *pSync) {                    \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                          \
        if (nvshm_use_tg_for_cpu_coll) {                                                         \
            call_alltoall##BITS##_on_stream_kern(dest, source, nelems, PE_start, logPE_stride,   \
                                                 PE_size, pSync, nvshmem_state->my_stream);      \
            CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));                           \
        } else {                                                                                 \
            nvshmemi_alltoall(dest, source, (BITS / 8), nelems, PE_start, logPE_stride, PE_size, \
                              pSync);                                                            \
        }                                                                                        \
    }

DEFN_NVSHMEM_CPU_ALLTOALL(32);
DEFN_NVSHMEM_CPU_ALLTOALL(64);
