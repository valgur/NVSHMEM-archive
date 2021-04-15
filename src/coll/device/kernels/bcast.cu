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

#define BCAST_ON_STREAM_KERNEL(TYPENAME, TYPE)                                              \
    __global__ void broadcast_##TYPENAME##_on_stream_kernel(                                \
        TYPE *dest, const TYPE *source, size_t nelems, int PE_root, int PE_start,           \
        int PE_stride, int PE_size, long *pSync) {                                          \
        if (!blockIdx.x)                                                                    \
            nvshmemxi_##TYPENAME##_broadcast_block(dest, source, nelems, PE_root, PE_start, \
                                                   PE_stride, PE_size, pSync);              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(BCAST_ON_STREAM_KERNEL)
#undef BCAST_ON_STREAM_KERNEL

#define CALL_BCAST_ON_STREAM(TYPENAME, TYPE)                                                       \
    extern "C" void call_##TYPENAME##_broadcast_on_stream_kern(                                    \
        TYPE *dest, const TYPE *source, size_t nelems, int PE_root, int PE_start,                  \
        int PE_stride, int PE_size, long *pSync, cudaStream_t stream) {                            \
        int num_threads_per_block = (MAX_THREADS_PER_CTA > nelems) ? nelems : MAX_THREADS_PER_CTA; \
        int num_blocks = 1;                                                                        \
        broadcast_##TYPENAME##_on_stream_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>( \
            dest, source, nelems, PE_root, PE_start, PE_stride, PE_size, pSync);                   \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                                    \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(CALL_BCAST_ON_STREAM)
#undef CALL_BCAST_ON_STREAM
