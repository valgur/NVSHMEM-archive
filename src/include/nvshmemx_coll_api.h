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

#ifndef _NVSHMEMX_COLL_API_H_
#define _NVSHMEMX_COLL_API_H_

#include "nvshmem_coll_common.h"

//==========================================
// nvshmem collective calls with stream param
//==========================================

// alltoall(s) collectives
void nvshmemx_alltoall32_on_stream(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_alltoall64_on_stream(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_alltoalls32_on_stream(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_alltoalls64_on_stream(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);

// barrier collectives
void nvshmemx_barrier_on_stream(NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_barrier_all_on_stream(cudaStream_t stream);

// sync collectives
void nvshmemx_sync_on_stream(NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_sync_all_on_stream(cudaStream_t stream);

// broadcast collectives
void nvshmemx_broadcast32_on_stream(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_broadcast64_on_stream(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET, cudaStream_t stream);

// collect and fcollect collectives
void nvshmemx_collect32_on_stream(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_collect64_on_stream(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_fcollect32_on_stream(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);
void nvshmemx_fcollect64_on_stream(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET, cudaStream_t stream);

// reduction collectives
#define NVSHMEMI_DECL_REDUCE_ONSTREAM(NAME, TYPE, OP)                           \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemx_##NAME##_##OP##_to_all_on_stream(  \
            TYPE *dest, const TYPE *src, int nreduce,                           \
            int PE_start, int logPE_stride, int PE_size,                        \
            TYPE *pWrk, long *pSync, cudaStream_t stream);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, prod)

#undef NVSHMEMI_DECL_REDUCE_ONSTREAM


//==========================================
// nvshmem collective calls on warp
//==========================================

// alltoall(s) collectives
__device__ void nvshmemx_alltoall32_warp(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_alltoall64_warp(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_alltoalls32_warp(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_alltoalls64_warp(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelem, NVSHMEMI_ASET);

// barrier collectives
__device__ void nvshmemx_barrier_warp(NVSHMEMI_ASET);
__device__ void nvshmemx_barrier_all_warp();

// sync collectives
__device__ void nvshmemx_sync_warp(NVSHMEMI_ASET);
__device__ void nvshmemx_sync_all_warp();

// broadcast collectives
__device__ void nvshmemx_broadcast32_warp(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET);
__device__ void nvshmemx_broadcast64_warp(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET);

// collect and fcollect collectives
__device__ void nvshmemx_collect32_warp(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_collect64_warp(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_fcollect32_warp(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_fcollect64_warp(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);

// reduction collectives
#define NVSHMEMI_DECL_REDUCE_ONWARP(NAME, TYPE, OP)                             \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemx_##NAME##_##OP##_to_all_warp(       \
            TYPE *dest, const TYPE *src, int nreduce,                           \
            int PE_start, int logPE_stride, int PE_size,                        \
            TYPE *pWrk, long *pSync);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONWARP, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONWARP, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONWARP, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONWARP, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONWARP, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONWARP, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONWARP, prod)

#undef NVSHMEMI_DECL_REDUCE_ONWARP

//==========================================
// nvshmem collective calls on block
//==========================================

// alltoall(s) collectives
__device__ void nvshmemx_alltoall32_block(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_alltoall64_block(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_alltoalls32_block(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_alltoalls64_block(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelem, NVSHMEMI_ASET);

// barrier collectives
__device__ void nvshmemx_barrier_block(NVSHMEMI_ASET);
__device__ void nvshmemx_barrier_all_block();

// barrier collectives
__device__ void nvshmemx_sync_block(NVSHMEMI_ASET);
__device__ void nvshmemx_sync_all_block();

// broadcast collectives
__device__ void nvshmemx_broadcast32_block(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET);
__device__ void nvshmemx_broadcast64_block(void *dest, const void *src, size_t nelem, int PE_root, NVSHMEMI_ASET);

// collect and fcollect collectives
__device__ void nvshmemx_collect32_block(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_collect64_block(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_fcollect32_block(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);
__device__ void nvshmemx_fcollect64_block(void *dest, const void *src, size_t nelem, NVSHMEMI_ASET);

// reduction collectives
#define NVSHMEMI_DECL_REDUCE_ONBLOCK(NAME, TYPE, OP)                            \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemx_##NAME##_##OP##_to_all_block(      \
            TYPE *dest, const TYPE *src, int nreduce,                           \
            int PE_start, int logPE_stride, int PE_size,                        \
            TYPE *pWrk, long *pSync);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONBLOCK, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONBLOCK, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONBLOCK, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONBLOCK, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONBLOCK, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONBLOCK, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONBLOCK, prod)

#undef NVSHMEMI_DECL_REDUCE_ONBLOCK

#endif /* NVSHMEMX_COLL_H */
