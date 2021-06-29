/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_util.h"
#include "nvshmemi_coll.h"
#include "gpu_coll.h"

#include "barrier_sync_algo.cuh"

#ifdef __CUDA_ARCH__

#define DEFN_NVSHMEMXI_BARRIER_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                                \
    __device__ void nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(int start, int stride, int size, \
                                                             long *pSync, long *counter) {    \
        int myIdx = nvshmemi_thread_id_in_##SC();                                             \
                                                                                              \
        NVSHMEMI_SYNC_##SC();                                                                 \
        if (!myIdx) nvshmem_quiet();                                                          \
        NVSHMEMI_SYNC_##SC();                                                                 \
                                                                                              \
        nvshmemi_sync_algo##SC_SUFFIX(start, stride, size, pSync, counter);                   \
                                                                                              \
        if (!myIdx) {                                                                         \
            if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY)            \
                nvshmemi_proxy_enforce_consistency_at_target(false);                          \
        }                                                                                     \
        NVSHMEMI_SYNC_##SC();                                                                 \
    }

DEFN_NVSHMEMXI_BARRIER_SCOPE(thread, , )
DEFN_NVSHMEMXI_BARRIER_SCOPE(warp, _warp, x)
DEFN_NVSHMEMXI_BARRIER_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMXI_BARRIER_SCOPE

#define DEFN_NVSHMEMX_BARRIER_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                              \
    __device__ int nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(nvshmem_team_t team) {          \
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];                  \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(teami->start, teami->stride, teami->size, \
                                                 nvshmemi_team_get_psync(teami, SYNC),     \
                                                 nvshmemi_team_get_sync_counter(teami));   \
        return 0;                                                                          \
    }

DEFN_NVSHMEMX_BARRIER_SCOPE(thread, , )
DEFN_NVSHMEMX_BARRIER_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_SCOPE

#define DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                          \
    __device__ void nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX() {                        \
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD];    \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(teami->start, teami->stride, teami->size, \
                                                 nvshmemi_team_get_psync(teami, SYNC),     \
                                                 nvshmemi_team_get_sync_counter(teami));   \
    }

DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(thread, , )
DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_ALL_SCOPE


#define DEFN_NVSHMEMXI_SYNC_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                                             \
    __device__ void nvshmem##SC_PREFIX##i_sync##SC_SUFFIX(int start, int stride, int size, long *pSync, long *counter) {  \
        int myidx = nvshmemi_thread_id_in_##SC();                                                       \
        NVSHMEMI_SYNC_##SC();                                                                           \
        nvshmemi_sync_algo##SC_SUFFIX(start, stride, size, pSync, counter);                             \
    }

DEFN_NVSHMEMXI_SYNC_SCOPE(thread, , )
DEFN_NVSHMEMXI_SYNC_SCOPE(warp, _warp, x)
DEFN_NVSHMEMXI_SYNC_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMXI_SYNC_SCOPE

#define DEFN_NVSHMEMX_SYNC_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                              \
    __device__ int nvshmem##SC_PREFIX##_team_sync##SC_SUFFIX(nvshmem_team_t team) {     \
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];               \
        nvshmem##SC_PREFIX##i_sync##SC_SUFFIX(teami->start, teami->stride, teami->size, \
                                              nvshmemi_team_get_psync(teami, SYNC),     \
                                              nvshmemi_team_get_sync_counter(teami));   \
        return 0;                                                                       \
    }

DEFN_NVSHMEMX_SYNC_SCOPE(thread, , )
DEFN_NVSHMEMX_SYNC_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_SCOPE

#define DEFN_NVSHMEMX_SYNC_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                          \
    __device__ void nvshmem##SC_PREFIX##_sync_all##SC_SUFFIX() {                        \
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD]; \
        nvshmem##SC_PREFIX##i_sync##SC_SUFFIX(teami->start, teami->stride, teami->size, \
                                              nvshmemi_team_get_psync(teami, SYNC),     \
                                              nvshmemi_team_get_sync_counter(teami));   \
    }

DEFN_NVSHMEMX_SYNC_ALL_SCOPE(thread, , )
DEFN_NVSHMEMX_SYNC_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_ALL_SCOPE

#endif
