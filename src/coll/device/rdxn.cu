/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_team.h"
#include "nvshmemi_coll.h"
#include "gpu_coll.h"

#ifdef __CUDA_ARCH__

template <typename TYPE, rdxn_ops_t OP>
__device__ inline void nvshmemi_reduce(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                       size_t nreduce) {
    nvshmemi_gpu_rdxn<TYPE, OP>(team, dest, source, nreduce);
}

#define DEFN_NVSHMEM_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                 \
    __device__ int nvshmem_##TYPENAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest,      \
                                                      const TYPE *source, size_t nreduce) { \
        nvshmemi_reduce<TYPE, RDXN_OPS_##OP>(team, dest, source, nreduce);                  \
        return 0;                                                                           \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, prod)

#undef DEFN_NVSHMEM_TYPENAME_OP_REDUCE
#endif
