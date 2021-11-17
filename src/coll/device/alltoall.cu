/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_coll.h"
#include "alltoall_device.cuh"
#include <cstdio>
#include <cassert>

#ifdef __CUDA_ARCH__

#define DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ int nvshmem##SC_PREFIX##_##TYPENAME##_alltoall##SC_SUFFIX(                     \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) {                 \
        nvshmemi_alltoall_threadgroup<TYPE, SC>(team, dest, source, nelems);                  \
        return 0;                                                                             \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP,
                                                 thread, , )
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, block, _block, x)
#undef DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP

#endif
