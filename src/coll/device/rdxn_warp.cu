/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
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

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, OP)  \
    __device__ int nvshmem##SC_PREFIX##_##TYPENAME##_##OP##_reduce##SC_SUFFIX(nvshmem_team_t team, TYPE *dest,      \
                                                            const TYPE *source, size_t nreduce) { \
        nvshmemi_reduce_threadgroup<TYPE, RDXN_OPS_##OP, nvshmemi_threadgroup_##SC>(team, dest, source, nreduce);        \
        return 0;                                                                                 \
    }

#define DEFN_NVSHMEM_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)                                            \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, and)    \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, or)     \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, xor)    \
                                                                                                             \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, max)   \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, min)   \
                                                                                                             \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, sum)      \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, SC_SUFFIX, SC_PREFIX, prod)

DEFN_NVSHMEM_REDUCE_THREADGROUP(warp, _warp, x);
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP
#undef DEFN_NVSHMEM_REDUCE_THREADGROUP

#endif
