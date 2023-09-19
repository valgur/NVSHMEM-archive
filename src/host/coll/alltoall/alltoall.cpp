/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "host/nvshmem_api.h"  // IWYU pragma: keep
#include <stdint.h>            // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-intn.h>
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <cuda_runtime.h>
#include <stddef.h>

#include "alltoall.h"
#include "common/nvshmem_common.cuh"
#include "internal/common/nvshmem_internal.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "common/nvshmem_types.h"
#include "internal/util.h"

#define DEFN_NVSHMEM_TYPENAME_ALLTOALL(TYPENAME, TYPE)                                            \
    int nvshmem_##TYPENAME##_alltoall(nvshmem_team_t team, TYPE *dest, const TYPE *source,        \
                                      size_t nelems) {                                            \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                           \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                        \
        nvshmemi_alltoall_on_stream<TYPE>(team, dest, source, nelems, nvshmemi_state->my_stream); \
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));                     \
        return 0;                                                                                 \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_ALLTOALL)
#undef DEFN_NVSHMEM_TYPENAME_ALLTOALL

int nvshmem_alltoallmem(nvshmem_team_t team, void *dest, const void *source, size_t nelems) {
    NVTX_FUNC_RANGE_IN_GROUP(COLL);
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();
    nvshmemi_alltoall_on_stream<char>(team, (char *)dest, (const char *)source, nelems,
                                      nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
    return 0;
}
