/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "host/nvshmem_api.h"  // IWYU pragma: keep
#include <cuda_runtime.h>

#include "barrier.h"
#include "host/nvshmem_api.h"
#include "internal/common/nvshmem_internal.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "common/nvshmem_common.cuh"
#include "common/nvshmem_types.h"
#include "internal/util.h"

void nvshmemi_barrier(nvshmem_team_t team) {
    nvshmem_quiet();
    nvshmemi_call_barrier_on_stream_kernel(team, nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
}

void nvshmemi_barrier_all() { nvshmemi_barrier(NVSHMEM_TEAM_WORLD); }

int nvshmem_barrier(nvshmem_team_t team) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();

    nvshmemi_barrier(team);

    return 0;
}

void nvshmem_barrier_all() {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    nvshmemi_check_state_and_init();
    nvshmemi_barrier_all();
    return;
}

void nvshmemi_sync(nvshmem_team_t team) {
    nvshmemi_call_sync_on_stream_kernel(team, nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
}

int nvshmem_team_sync(nvshmem_team_t team) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();

    nvshmemi_sync(team);

    return 0;
}

void nvshmem_sync_all() {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    nvshmemi_check_state_and_init();

    nvshmemxi_sync_all_on_stream(nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
}
