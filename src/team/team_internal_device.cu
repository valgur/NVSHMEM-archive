/*
* Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
*
* See COPYRIGHT for license information
*/

#define NVSHMEMI_DEVICE_ONLY
#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_coll.h"
#include "nvshmem_internal.h"
#include "nvshmemi_team.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "util.h"
#include "gpu_coll.h"
#include "team_internal.h"

#ifdef __CUDA_ARCH__
__device__ int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team) {
    int src_pe_world, dest_pe = -1;

    if (src_pe > src_team->size) return -1;

    src_pe_world = src_team->start + src_pe * src_team->stride;
    assert(src_pe_world >= src_team->start && src_pe_world < nvshmemi_device_state_d.npes);

    dest_pe = nvshmemi_pe_in_active_set(src_pe_world, dest_team->start, dest_team->stride,
                                        dest_team->size);

    return dest_pe;
}


__device__ long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op) {
    long *team_psync;
    team_psync = &nvshmemi_device_state_d.psync_pool[team->team_idx * get_psync_len_per_team()];
    switch(op) {
        case SYNC:
            return team_psync;
        case REDUCE:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE + (NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * (team->rdxn_count % 2))];
        case BCAST:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE + 2 * NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE];
        case FCOLLECT:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE + 2 * NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE + NVSHMEMI_BCAST_SYNC_SIZE];
        default:
            printf("Incorrect argument to nvshmemi_team_get_psync\n");
            return NULL;
    }
}

__device__ long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team) {
    return &nvshmemi_device_state_d.sync_counter[2 * team->team_idx];
}
#endif
