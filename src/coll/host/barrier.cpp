/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "cpu_coll.h"

void nvshmemi_sync(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int stride = 1 << logPE_stride;
    int PE_end = PE_start + (stride * PE_size);
    int root = PE_start;
    int val = 1;
    volatile int *bcast_sync_arr = nvshm_cpu_coll_info.cpu_bcast_int_sync_arr;

    if (root == nvshmem_state->mype) {
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
            }
        }

        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            bcast_sync_arr[ii] = val;
        }

        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
            }
        }

    } else {
        while (val != bcast_sync_arr[nvshmem_state->mype]) {
        }

        bcast_sync_arr[nvshmem_state->mype] = 0;
    }
}

void nvshmemi_sync_p2p(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    int stride = 1 << logPE_stride;
    int PE_end = PE_start + (stride * PE_size);
    int root = PE_start;
    int val = 1;
    volatile int *gpu_bcast_sync_arr = nvshm_cpu_coll_info.gpu_bcast_int_sync_arr;

    if (root == nvshmem_state->mype) {
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            while (0 != gpu_bcast_sync_arr[ii]) {
            }
        }

        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            nvshmem_int_p((int *)&gpu_bcast_sync_arr[ii], val, ii);
        }

        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            while (0 != gpu_bcast_sync_arr[ii]) {
            }
        }

    } else {
        while (val != gpu_bcast_sync_arr[nvshmem_state->mype]) {
        }

        nvshmem_int_p((int *)&gpu_bcast_sync_arr[nvshmem_state->mype], 0, 0);
    }
}

void nvshmemi_barrier_shm(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmem_quiet();
    if (!nvshm_enable_p2p_cpu_coll) {
        nvshmemi_sync(PE_start, logPE_stride, PE_size, pSync);
    } else {
        nvshmemi_sync_p2p(PE_start, logPE_stride, PE_size, pSync);
    }
}

void nvshmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    if (nvshm_use_tg_for_cpu_coll) {
        call_barrier_on_stream_kern(PE_start, logPE_stride, PE_size, pSync,
                                    nvshmem_state->my_stream);
        CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));
    } else {
        nvshmemi_barrier_shm(PE_start, logPE_stride, PE_size, pSync);
    }
}

void nvshmemi_barrier_all_shm() {
    int PE_start = 0;
    int logPE_stride = 0;
    int PE_size = nvshmem_state->npes;

    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmem_quiet();
    if (!nvshm_enable_p2p_cpu_coll) {
        nvshmemi_sync(PE_start, logPE_stride, PE_size, NULL);
    } else {
        nvshmemi_sync_p2p(PE_start, logPE_stride, PE_size, NULL);
    }
}

void nvshmemi_barrier_all() {
    static int first_time = 1;
    nvshmem_quiet();  // host-side quiet is needed

    if (nvshm_use_tg_for_cpu_coll) {
        call_barrier_all_on_stream_kern(nvshmem_state->my_stream);
        TRACE(NVSHMEM_COLL, "In nvshmemi_barrier_all, calling cuStreamSynchronize");
        CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));
    } else {
        nvshmemi_barrier_all_shm();
    }

    return;
}

void nvshmemi_sync_all() {
    int PE_start = 0;
    int logPE_stride = 0;
    int PE_size = nvshmem_state->npes;

    if (nvshm_use_tg_for_cpu_coll) {
        call_sync_all_on_stream_kern(nvshmem_state->my_stream);
        CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));
    } else {
        if (!nvshm_enable_p2p_cpu_coll) {
            nvshmemi_sync(PE_start, logPE_stride, PE_size, NULL);
        } else {
            nvshmemi_sync_p2p(PE_start, logPE_stride, PE_size, NULL);
        }
    }

    return;
}

void nvshmem_barrier_all() {
    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmemi_barrier_all();

    return;
}

void nvshmem_sync(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    if (nvshm_use_tg_for_cpu_coll) {
        call_sync_on_stream_kern(PE_start, logPE_stride, PE_size, pSync, nvshmem_state->my_stream);
        CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));
    } else {
        if (!nvshm_enable_p2p_cpu_coll) {
            nvshmemi_sync(PE_start, logPE_stride, PE_size, NULL);
        } else {
            nvshmemi_sync_p2p(PE_start, logPE_stride, PE_size, NULL);
        }
    }
}

void nvshmem_sync_all() {
    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmemi_sync_all();

    return;
}
