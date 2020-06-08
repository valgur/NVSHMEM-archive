/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_COLL_API_H_
#define _NVSHMEM_COLL_API_H_

#include "nvshmem_coll_common.h"

//===============================
// standard nvshmem collective calls
//===============================

// alltoall(s) collectives
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_alltoall32(void *dest, const void *src, size_t nelems, NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_alltoall64(void *dest, const void *src, size_t nelems, NVSHMEMI_ASET);

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_alltoalls32(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_alltoalls64(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, NVSHMEMI_ASET);

// barrier collectives
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_barrier(NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_barrier_all();

// sync collectives
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_sync(NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_sync_all();

// broadcast collectives
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_broadcast32(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_broadcast64(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET);

// collect and fcollect collectives
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_collect32(void *dest, const void *src, size_t nelems, NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_collect64(void *dest, const void *src, size_t nelems, NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_fcollect32(void *dest, const void *src, size_t nelems, NVSHMEMI_ASET);
NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_fcollect64(void *dest, const void *src, size_t nelems, NVSHMEMI_ASET);

// reduction collectives
#define NVSHMEMI_DECL_REDUCE(NAME, TYPE, OP)                            \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmem_##NAME##_##OP##_to_all(     \
            TYPE *dest, const TYPE *src, int nreduce,                   \
            int PE_start, int logPE_stride, int PE_size, TYPE *pWrk, long *pSync);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE, prod)

#undef NVSHMEMI_DECL_REDUCE

#endif /* NVSHMEM_COLL_H */
