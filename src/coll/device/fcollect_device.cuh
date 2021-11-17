/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef FCOLLECT_DEVICE_CUH
#define FCOLLECT_DEVICE_CUH
#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_coll.h"
#include <cstdio>
#include <cassert>

#ifdef __CUDA_ARCH__

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                             const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int next_rank;
    int next_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    //nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        next_offset = nelems * ((mype - PE_start) / stride);
        nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest + next_offset, source, nelems, next_rank);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_p2p_allpush_threadgroup(nvshmem_team_t team, T *dest,
                                                                 const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int next_rank;
    int next_offset;
    const int mype = nvshmemi_device_state_d.mype;
    int my_idx_in_active_set = (mype - PE_start) / PE_stride;
    T *dst_ptr;
    nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = 0; ii < PE_size; ii++) {
        next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;
        next_offset = nelems * my_idx_in_active_set;
        dst_ptr = (T *)nvshmem_ptr((void *)(dest + next_offset), next_rank);
        nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source, nelems * sizeof(T));
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_fcollect_threadgroup(nvshmem_team_t team, T *dest, const T *source,
                                                     size_t nelems) {
    if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST)
        nvshmemi_fcollect_p2p_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
    else {
        nvshmemi_fcollect_allpush_threadgroup<T, SCOPE>(team, dest, source, nelems);
    }
}

#endif
#endif /* FCOLLECT_DEVICE_CUH */
