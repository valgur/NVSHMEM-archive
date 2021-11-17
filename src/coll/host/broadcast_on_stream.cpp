/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_nvtx.hpp"
#include "cpu_coll.h"

#define DEFN_NVSHMEMX_BROADCAST_ON_STREAM(TYPENAME, TYPE)                                         \
    int nvshmemx_##TYPENAME##_broadcast_on_stream(nvshmem_team_t team, TYPE *dest,                \
                                                  const TYPE *source, size_t nelems, int PE_root, \
                                                  cudaStream_t stream) {                          \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                           \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                           \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                        \
        nvshmemi_team_t *teami = nvshmemi_team_pool[team];                                        \
        if (nvshmemi_use_nccl && NCCL_DT_##TYPENAME != -1) {                                      \
            NCCL_CHECK(nccl_ftable.Broadcast(source, dest, nelems,                                \
                                             (ncclDataType_t)NCCL_DT_##TYPENAME, PE_root,         \
                                             teami->nccl_comm, stream));                          \
        } else {                                                                                  \
            nvshmemi_call_broadcast_on_stream_kernel<TYPE>(team, dest, source, nelems, PE_root,   \
                                                           stream);                               \
        }                                                                                         \
        return 0;                                                                                 \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_BROADCAST_ON_STREAM)
#undef DEFN_NVSHMEMX_BROADCAST_ON_STREAM
