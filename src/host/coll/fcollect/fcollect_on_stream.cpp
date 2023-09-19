/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "host/nvshmemx_api.h"  // IWYU pragma: keep
#include <driver_types.h>
#include <stddef.h>

#include "fcollect.h"
#include "common/nvshmem_common.cuh"
#include "internal/common/nvshmem_internal.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "common/nvshmem_types.h"
#include "internal/util.h"

#define DEFN_NVSHMEMX_TYPENAME_FCOLLECT_ON_STREAM(TYPENAME, TYPE)                                  \
    int nvshmemx_##TYPENAME##_fcollect_on_stream(                                                  \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, cudaStream_t stream) { \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                            \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        return nvshmemi_fcollect_on_stream<TYPE>(team, dest, source, nelems, stream);              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_TYPENAME_FCOLLECT_ON_STREAM)
