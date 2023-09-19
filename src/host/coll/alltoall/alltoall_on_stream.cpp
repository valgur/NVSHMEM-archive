/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "host/nvshmemx_api.h"  // IWYU pragma: keep
#include <stdint.h>             // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-intn.h>
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <driver_types.h>
#include <stddef.h>

#include "alltoall.h"
#include "common/nvshmem_common.cuh"
#include "internal/common/nvshmem_internal.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "common/nvshmem_types.h"
#include "internal/util.h"

#define DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM(TYPENAME, TYPE)                                  \
    int nvshmemx_##TYPENAME##_alltoall_on_stream(                                                  \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, cudaStream_t stream) { \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                            \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        return nvshmemi_alltoall_on_stream<TYPE>(team, dest, source, nelems, stream);              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM)
#undef DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM

int nvshmemx_alltoallmem_on_stream(nvshmem_team_t team, void *dest, const void *source,
                                   size_t nelems, cudaStream_t stream) {
    return nvshmemx_char_alltoall_on_stream(team, (char *)dest, (const char *)source, nelems,
                                            stream);
}
