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


#ifdef __CUDA_ARCH__

#define DEFN_NVSHMEMX_BARRIER_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                     \
    __device__ int nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(nvshmem_team_t team) { \
        nvshmemi_barrier_threadgroup<SC>(team);                                   \
        return 0;                                                                 \
    }

DEFN_NVSHMEMX_BARRIER_SCOPE(thread, , )
DEFN_NVSHMEMX_BARRIER_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_SCOPE

#define DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)   \
    __device__ void nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX() { \
        nvshmemi_barrier_threadgroup<SC>(NVSHMEM_TEAM_WORLD);       \
    }

DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(thread, , )
DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_ALL_SCOPE

#define DEFN_NVSHMEMX_SYNC_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                          \
    __device__ int nvshmem##SC_PREFIX##_team_sync##SC_SUFFIX(nvshmem_team_t team) { \
        nvshmemi_sync_threadgroup<SC>(team);                                        \
        return 0;                                                                   \
    }

DEFN_NVSHMEMX_SYNC_SCOPE(thread, , )
DEFN_NVSHMEMX_SYNC_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_SCOPE

#define DEFN_NVSHMEMX_SYNC_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)   \
    __device__ void nvshmem##SC_PREFIX##_sync_all##SC_SUFFIX() { \
        nvshmemi_sync_threadgroup<SC>(NVSHMEM_TEAM_WORLD);       \
    }

DEFN_NVSHMEMX_SYNC_ALL_SCOPE(thread, , )
DEFN_NVSHMEMX_SYNC_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_ALL_SCOPE

#endif
