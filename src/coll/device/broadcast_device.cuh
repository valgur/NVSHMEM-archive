/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef BROADCAST_DEVICE_CUH
#define BROADCAST_DEVICE_CUH

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_coll.h"
#include <cstdio>
#include <iostream>
#include <cassert>

#ifdef __CUDA_ARCH__

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_put2all_threadgroup(nvshmem_team_t team, T *dest,
                                                         const T *source, size_t nelems,
                                                         int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int i;
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int root = nvshmemi_team_translate_pe(teami, PE_root, &nvshmemi_device_state_d.team_world);
    int PE_end = PE_start + (stride * PE_size);
    if (root == nvshmemi_device_state_d.mype) {
        for (i = PE_start; i < PE_end; i += stride) {
            nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest, source, nelems, i);
        }
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_put2all_direct_threadgroup(nvshmem_team_t team, T *dest,
                                                                const T *source, size_t nelems,
                                                                int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int i;
    int PE_start = teami->start;
    int PE_stride = teami->stride;
    int PE_size = teami->size;
    int stride = PE_stride;
    int root = nvshmemi_team_translate_pe(teami, PE_root, &nvshmemi_device_state_d.team_world);
    int PE_end = PE_start + (stride * PE_size);
    T *dst_ptr;
    if (root == nvshmemi_device_state_d.mype) {
        for (i = PE_start; i < PE_end; i += stride) {
            dst_ptr = (T *)nvshmem_ptr(dest, i);
            nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source, nelems * sizeof(T));
        }
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_broadcast_threadgroup(nvshmem_team_t team, T *dest, const T *source,
                                                      size_t nelems, int PE_root) {
    if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST)
        nvshmemi_bcast_put2all_direct_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
    else
        nvshmemi_bcast_put2all_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
}

#endif /* __CUDA_ARCH__ */

#endif
