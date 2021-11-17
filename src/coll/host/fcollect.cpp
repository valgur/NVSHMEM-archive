/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_nvtx.hpp"
#include "cpu_coll.h"

#define DEFN_NVSHMEM_TYPENAME_FCOLLECT(TYPENAME, TYPE)                                             \
    int nvshmem_##TYPENAME##_fcollect(nvshmem_team_t team, TYPE *dest, const TYPE *source,         \
                                      size_t nelems) {                                             \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                            \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                         \
        nvshmemi_team_t *teami = nvshmemi_team_pool[team];                                         \
        if (nvshmemi_use_nccl && NCCL_DT_##TYPENAME != -1) {                                       \
            NCCL_CHECK(nccl_ftable.AllGather(source, dest, nelems,                                 \
                                             (ncclDataType_t)NCCL_DT_##TYPENAME, teami->nccl_comm, \
                                             nvshmemi_state->my_stream));                          \
        } else {                                                                                   \
            nvshmemi_call_fcollect_on_stream_kernel<TYPE>(team, dest, source, nelems,              \
                                                          nvshmemi_state->my_stream);              \
        }                                                                                          \
        CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));                                \
        return 0;                                                                                  \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_FCOLLECT)
