/****
 * Copyright (c) 2016-2019, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#ifndef __DEVICE_UTIL_H
#define __DEVICE_UTIL_H

#ifdef __CUDA_ARCH__
__device__ inline int nvshmemi_thread_id_in_warp() {
    int myIdx;
    asm volatile("mov.u32  %0, %laneid;" : "=r"(myIdx));
    return myIdx;
}

__device__ inline int nvshmemi_warp_size() {
    return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
               ? (blockDim.x * blockDim.y * blockDim.z)
               : warpSize;
}

__device__ inline int nvshmemi_thread_id_in_block() {
    return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
}

__device__ inline int nvshmemi_block_size() { return (blockDim.x * blockDim.y * blockDim.z); }

__device__ inline void nvshmemi_warp_sync() { __syncwarp(); }

__device__ inline void nvshmemi_block_sync() { __syncthreads(); }
#endif

#endif
