/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemi_coll.h"
#include "nvshmem_nvtx.hpp"
#include "cpu_coll.h"

#define DEFN_NVSHMEMI_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                      \
    void nvshmemi_##TYPENAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest, const TYPE *source, \
                                             size_t nreduce) {                                    \
        nvshmemi_call_rdxn_on_stream_kernel<TYPE, RDXN_OPS_##OP>(team, dest, source, nreduce,     \
                                                                 nvshmemi_state->my_stream);      \
        CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));                               \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, prod)

#define DEFN_NVSHMEM_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                     \
    int nvshmem_##TYPENAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest, const TYPE *source, \
                                           size_t nreduce) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(COLL);                                                         \
        NVSHMEMI_CHECK_INIT_STATUS();                                                           \
        NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();                                      \
        nvshmemi_team_t *teami = nvshmemi_team_pool[team];                                      \
        if (nvshmemi_use_nccl && NCCL_REDOP_##OP != -1 && NCCL_DT_##TYPENAME != -1) {           \
            NCCL_CHECK(nccl_ftable.AllReduce(                                                   \
                source, dest, nreduce, (ncclDataType_t)NCCL_DT_##TYPENAME,                      \
                (ncclRedOp_t)NCCL_REDOP_##OP, teami->nccl_comm, nvshmemi_state->my_stream));    \
            CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));                         \
        } else {                                                                                \
            nvshmemi_##TYPENAME##_##OP##_reduce(team, dest, source, nreduce);                   \
        }                                                                                       \
        return 0;                                                                               \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, prod)
