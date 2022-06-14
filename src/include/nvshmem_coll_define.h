/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_COLL_DEFINE_H_
#define _NVSHMEM_COLL_DEFINE_H_

#ifdef __CUDA_ARCH__

// alltoall(s) collectives
#define DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE(SCOPE, TYPENAME, TYPE) \
    __device__ int nvshmemx_##TYPENAME##_alltoall_##SCOPE(nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nelem);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE, block)
#undef DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE

__device__ int nvshmemx_alltoallmem_warp(nvshmem_team_t team, void *dest, const void *src, size_t nelem);
__device__ int nvshmemx_alltoallmem_block(nvshmem_team_t team, void *dest, const void *src, size_t nelem);

#endif /* __CUDA_ARCH__ */
#endif /* NVSHMEMX_COLL_DEFINE_H */
