/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"

#include "nvshmem_internal.h"
#include "util.h"
#include "nvshmemi_team.h"

/* Team Managment Routines */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDA_ARCH__

__device__ int nvshmem_team_my_pe(nvshmem_team_t team) {
    if (team == NVSHMEM_TEAM_INVALID)
        return -1;
    else if (team == NVSHMEM_TEAM_WORLD)
        return nvshmemi_device_state_d.mype;
    else if (team == NVSHMEMX_TEAM_NODE)
        return nvshmemi_device_state_d.node_mype;
    else
        return nvshmemi_device_state_d.team_pool[team]->my_pe;
}

__device__ int nvshmem_team_n_pes(nvshmem_team_t team) {
    if (team == NVSHMEM_TEAM_INVALID)
        return -1;
    else if (team == NVSHMEM_TEAM_WORLD)
        return nvshmemi_device_state_d.npes;
    else if (team == NVSHMEMX_TEAM_NODE)
        return nvshmemi_device_state_d.node_npes;
    else
        return nvshmemi_device_state_d.team_pool[team]->size;
}

__device__ int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe,
                                         nvshmem_team_t dest_team) {
    if (src_team == NVSHMEM_TEAM_INVALID || dest_team == NVSHMEM_TEAM_INVALID) return -1;
    nvshmemi_team_t *src_teami, *dest_teami;
    src_teami = nvshmemi_device_state_d.team_pool[src_team];
    dest_teami = nvshmemi_device_state_d.team_pool[dest_team];

    return nvshmemi_team_translate_pe(src_teami, src_pe, dest_teami);
}

#endif

#ifdef __cplusplus
}
#endif
