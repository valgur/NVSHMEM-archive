/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_nvtx.hpp"
#include "cpu_coll.h"

#define DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM(TYPENAME, TYPE)                                  \
    int nvshmemx_##TYPENAME##_alltoall_on_stream(                                                  \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, cudaStream_t stream) { \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                            \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                            \
        nvshmemi_team_t *teami = nvshmemi_team_pool[team];                                         \
        int team_n_pes = nvshmem_team_n_pes(team);                                                 \
        if (nvshmemi_use_nccl && NCCL_DT_##TYPENAME != -1 &&                                       \
            ((nccl_version >= 2700 && team_n_pes <= 4096 /* NCCL limit for Group API */) ||        \
             (nccl_version >= 2800 && team_n_pes <= 32768 /* NCCL limit for Group API */))) {      \
            size_t rank_offset = nelems * sizeof(TYPE);                                            \
            NCCL_CHECK(nccl_ftable.GroupStart());                                                  \
            for (int pe = 0; pe < team_n_pes; pe++) {                                              \
                NCCL_CHECK(nccl_ftable.Send(((char *)source) + pe * rank_offset, nelems,           \
                                            (ncclDataType_t)NCCL_DT_##TYPENAME, pe,                \
                                            teami->nccl_comm, stream));                            \
                NCCL_CHECK(nccl_ftable.Recv(((char *)dest) + pe * rank_offset, nelems,             \
                                            (ncclDataType_t)NCCL_DT_##TYPENAME, pe,                \
                                            teami->nccl_comm, stream));                            \
            }                                                                                      \
            NCCL_CHECK(nccl_ftable.GroupEnd());                                                    \
        } else {                                                                                   \
            nvshmemx_barrier_on_stream(team, stream);                                              \
            call_##TYPENAME##_alltoall_on_stream_kern(                                             \
                dest, source, nelems, teami->start, teami->stride, teami->size,                    \
                nvshmemi_team_get_psync(teami, ALLTOALL), stream);                                 \
        }                                                                                          \
        return 0;                                                                                  \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM)
#undef DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM
