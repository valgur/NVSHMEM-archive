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
#include "prims_ll.h"
#include <cstdio>
#include <iostream>
#include <cassert>

#ifdef __CUDA_ARCH__

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_bcast_tree_threadgroup(nvshmem_team_t team, T *dest,
                                                       const T *source, size_t nelems,
                                                       int PE_root) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const size_t bcast_ll_threshold = nvshmemi_device_state_d.bcast_ll_threshold;
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) /* Only one thread should increment */
        teami->ll_flag++;
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, BCAST);
    const size_t bcast_count = teami->bcast_count;
    size_t recv_offset = bcast_count * bcast_ll_threshold * 2;
    const int my_pe_in_team = nvshmem_team_my_pe(team);
    const int k = 3;

    if (recv_offset + bcast_ll_threshold * 2 > sizeof(long) * NVSHMEMI_BCAST_SYNC_SIZE) {
        nvshmemi_barrier_threadgroup<SCOPE>(team);
        recv_offset = 0;
        teami->bcast_count = 0;
    }

    if (PE_root != my_pe_in_team) {
        nvshmemi_recvLL<T, SCOPE>(dest, (uint64_t *)(pWrk + recv_offset), nelems, ll_flag);
    } else {
        nvshmemi_packLL<T, SCOPE>((uint64_t *)(pWrk + recv_offset), source, nelems, ll_flag);
    }
    for (int i = 0; i < k; i++) {
        int child_in_team = (my_pe_in_team * k + i + 1);
        if (child_in_team >= teami->size) break;
        int child = nvshmemi_team_translate_pe(
            teami, child_in_team, nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD_INDEX]);

        nvshmemi_put_nbi_threadgroup<uint64_t, SCOPE>((uint64_t *)(pWrk + recv_offset),
                                                      (uint64_t *)(pWrk + recv_offset),
                                                      nelems * sizeof(T) / sizeof(uint32_t), child);
    }
    if (PE_root == my_pe_in_team)
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nelems * sizeof(T));
    nvshmemi_threadgroup_sync<SCOPE>();
    if (!myIdx) __threadfence();
    nvshmemi_threadgroup_sync<SCOPE>();
}

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
    int root = nvshmemi_team_translate_pe(
        teami, PE_root, nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD_INDEX]);
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
    int root = nvshmemi_team_translate_pe(
        teami, PE_root, nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD_INDEX]);
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
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    if (!myIdx) /* Only one thread should increment bcast_count */
        nvshmemi_device_state_d.team_pool[team]->bcast_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
    if (nvshmemi_device_state_d.bcast_ll_threshold >= nelems && sizeof(T) >= sizeof(uint32_t) &&
        nelems % 2 == 0) {
        nvshmemi_bcast_tree_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
    } else if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS) {
        nvshmemi_bcast_put2all_direct_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
    } else {
        nvshmemi_bcast_put2all_threadgroup<T, SCOPE>(team, dest, source, nelems, PE_root);
    }
}

#endif /* __CUDA_ARCH__ */

#endif
