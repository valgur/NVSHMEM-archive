/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_coll.h"
#include <cstdio>
#include <cassert>
#include "rdxn_device.cuh"

#ifdef __CUDA_ARCH__

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP)                      \
    __device__ int nvshmemx_##TYPENAME##_##OP##_reduce_##SC(nvshmem_team_t team, TYPE *dest,      \
                                                            const TYPE *source, size_t nreduce) { \
        nvshmemi_reduce_threadgroup<TYPE, RDXN_OPS_##OP, SC>(team, dest, source, nreduce);        \
        return 0;                                                                                 \
    }

#define DEFN_NVSHMEM_REDUCE_THREADGROUP(SC)                                                                 \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, and)    \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, or)     \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, xor)    \
                                                                                                            \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, max)   \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, min)   \
                                                                                                            \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, sum)      \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, prod)

DEFN_NVSHMEM_REDUCE_THREADGROUP(warp);
DEFN_NVSHMEM_REDUCE_THREADGROUP(block);
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP
#undef DEFN_NVSHMEM_REDUCE_THREADGROUP

#endif
