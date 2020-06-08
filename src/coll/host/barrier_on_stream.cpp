/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"

void nvshmemxi_sync_p2p_on_stream(int PE_start, int logPE_stride, int PE_size,
                                  cudaStream_t stream) {
    int stride = 1 << logPE_stride;
    int PE_end = PE_start + (stride * PE_size);
    int root = PE_start;
    int val = 1;
    int *wait_ptr;
    int wait_val = 0;
    int wait_flag = CU_STREAM_WAIT_VALUE_EQ;
    int *write_ptr;
    int write_val = 0;
    int write_flag = CU_STREAM_WRITE_VALUE_DEFAULT;
    volatile int *gpu_bcast_sync_arr = nvshm_cpu_coll_info.gpu_bcast_int_sync_arr;

    if (root == nvshmem_state->mype) {
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            wait_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            wait_val = 1;
            CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));

            // reset so that stale value isn't read next time
            write_val = 0;
            write_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        }

        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (root == ii) continue;
            nvshmemx_int_p_on_stream((int *)&gpu_bcast_sync_arr[ii], val, ii, stream);
        }

    } else {
        nvshmemx_int_p_on_stream((int *)&gpu_bcast_sync_arr[nvshmem_state->mype], 1, 0, stream);

        wait_ptr = (int *)&(gpu_bcast_sync_arr[nvshmem_state->mype]);
        wait_val = val;
        CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));

        nvshmemx_int_p_on_stream((int *)&gpu_bcast_sync_arr[nvshmem_state->mype], 0, 0, stream);
    }
}

int nvshmemxi_barrier_cpu_base_on_stream(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                         cudaStream_t stream) {
    int status = 0;
    int *wait_ptr;
    int wait_val = 0;
    int wait_flag = CU_STREAM_WAIT_VALUE_EQ;
    int *write_ptr;
    int write_val = 0;
    int write_flag = CU_STREAM_WRITE_VALUE_DEFAULT;
    int PE_root = PE_start;
    int stride = 1 << logPE_stride;
    int ii;
    int offset;
    char *base = NULL;
    volatile int *gpu_bcast_sync_arr = nvshm_cpu_coll_info.gpu_bcast_int_sync_arr;
    volatile int *peer_gpu_bcast_sync_arr = NULL;

    base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
    offset = (char *)gpu_bcast_sync_arr - base;

    // global barrier among PEs using gpu nvshmem

    if (PE_root == nvshmem_state->mype) {
        // wait for all non-root streams to arrive at barrier
        for (ii = PE_start; ii <= (PE_start + (stride * (PE_size - 1))); ii++) {
            if (PE_root == ii) continue;
            wait_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            wait_val = 1;
            CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
            write_val = 0;
            write_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        }

        // notify non-root streams that all processes are in barrier
        for (ii = PE_start; ii <= (PE_start + (stride * (PE_size - 1))); ii++) {
            if (PE_root == ii) continue;
            peer_gpu_bcast_sync_arr =
                (int *)((char *)nvshmem_state->peer_heap_base[nvshmem_state->mype] + offset);
            write_val = 0;
            write_ptr = (int *)&(peer_gpu_bcast_sync_arr[ii]);
            CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        }

    } else {
        // stream notifies root stream of arrival
        peer_gpu_bcast_sync_arr = (int *)((char *)nvshmem_state->peer_heap_base[0] + offset);
        write_val = 1;
        write_ptr = (int *)&(peer_gpu_bcast_sync_arr[nvshmem_state->mype]);
        CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        // stream waits for root stream to notify arrival of all processes
        wait_ptr = (int *)&(gpu_bcast_sync_arr[nvshmem_state->mype]);
        wait_val = 0;
        CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
    }
fn_out:
    return status;
}

int nvshmemxi_barrier_all_cpu_base_on_stream(cudaStream_t stream) {
    int status = 0;
    int *wait_ptr;
    int wait_val = 0;
    int wait_flag = CU_STREAM_WAIT_VALUE_EQ;
    int *write_ptr;
    int write_val = 0;
    int write_flag = CU_STREAM_WRITE_VALUE_DEFAULT;
    int PE_root = 0;
    int offset;
    char *base = NULL;
    volatile int *gpu_bcast_sync_arr = nvshm_cpu_coll_info.gpu_bcast_int_sync_arr;
    volatile int *peer_gpu_bcast_sync_arr = NULL;

    base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
    offset = (char *)gpu_bcast_sync_arr - base;

    // global barrier among PEs using gpu nvshmem

    if (PE_root == nvshmem_state->mype) {
        // wait for all non-root streams to arrive at barrier_all
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (PE_root == ii) continue;
            wait_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            wait_val = 1;
            CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
            write_val = 0;
            write_ptr = (int *)&(gpu_bcast_sync_arr[ii]);
            CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        }

        // notify non-root streams that all processes are in barrier_all
        for (int ii = 0; ii < nvshmem_state->npes; ii++) {
            if (PE_root == ii) continue;
            peer_gpu_bcast_sync_arr =
                (int *)((char *)nvshmem_state->peer_heap_base[nvshmem_state->mype] + offset);
            write_val = 0;
            write_ptr = (int *)&(peer_gpu_bcast_sync_arr[ii]);
            CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));
        }
    } else {
        // stream notifies root stream of arrival
        peer_gpu_bcast_sync_arr = (int *)((char *)nvshmem_state->peer_heap_base[0] + offset);
        write_val = 1;
        write_ptr = (int *)&(peer_gpu_bcast_sync_arr[nvshmem_state->mype]);
        CUDA_CHECK(cuStreamWriteValue32(stream, (CUdeviceptr)write_ptr, write_val, write_flag));

        // stream waits for root stream to notify arrival of all processes
        wait_ptr = (int *)&(gpu_bcast_sync_arr[nvshmem_state->mype]);
        wait_val = 0;
        CUDA_CHECK(cuStreamWaitValue32(stream, (CUdeviceptr)wait_ptr, wait_val, wait_flag));
    }

fn_out:
    return status;
}

void nvshmemx_barrier_all_on_stream(cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    // quiet or quiet on stream missing here?

    if (nvshm_use_tg_for_stream_coll) {
        call_barrier_all_on_stream_kern(stream);
    } else {
        if (!nvshm_enable_p2p_cpu_coll) {
            nvshmemxi_barrier_all_cpu_base_on_stream(stream);
        } else {
            nvshmemxi_sync_p2p_on_stream(0, 0, nvshmem_state->npes, stream);
        }
    }
    return;
}

void nvshmemx_barrier_on_stream(int PE_start, int logPE_stride, int PE_size, long *pSync,
                                cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    // quiet or quiet on stream missing here?

    if (nvshm_use_tg_for_stream_coll) {
        call_barrier_on_stream_kern(PE_start, logPE_stride, PE_size, pSync, stream);
    } else {
        if (!nvshm_enable_p2p_cpu_coll) {
            nvshmemxi_barrier_cpu_base_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
        } else {
            nvshmemxi_sync_p2p_on_stream(PE_start, logPE_stride, PE_size, stream);
        }
    }
    return;
}

void nvshmemx_sync_all_on_stream(cudaStream_t stream) {
    int PE_start = 0;
    int logPE_stride = 0;
    int PE_size = nvshmem_state->npes;

    NVSHMEM_CHECK_STATE_AND_INIT();

    if (nvshm_use_tg_for_stream_coll) {
        call_sync_all_on_stream_kern(stream);
    } else {
        if (!nvshm_enable_p2p_cpu_coll) {
            nvshmemxi_barrier_cpu_base_on_stream(PE_start, logPE_stride, PE_size, NULL, stream);
        } else {
            nvshmemxi_sync_p2p_on_stream(PE_start, logPE_stride, PE_size, stream);
        }
    }
    return;
}

void nvshmemx_sync_on_stream(int PE_start, int logPE_stride, int PE_size, long *pSync,
                             cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    if (nvshm_use_tg_for_stream_coll) {
        call_sync_on_stream_kern(PE_start, logPE_stride, PE_size, pSync, stream);
    } else {
        if (!nvshm_enable_p2p_cpu_coll) {
            nvshmemxi_barrier_cpu_base_on_stream(PE_start, logPE_stride, PE_size, pSync, stream);
        } else {
            nvshmemxi_sync_p2p_on_stream(PE_start, logPE_stride, PE_size, stream);
        }
    }
    return;
}
