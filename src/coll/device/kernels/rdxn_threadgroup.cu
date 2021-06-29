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

#define RDXN_ON_STREAM_KERNEL(TYPENAME, TYPE, OP)                                                \
    __global__ void rdxn_##TYPENAME##_##OP##_on_stream_kernel(                                   \
        TYPE *dest, const TYPE *source, size_t nreduce, int start, int stride, int size,         \
        uint64_t *pWrk, uint64_t *pSync) {                                                       \
        if (!blockIdx.x)                                                                         \
            nvshmemxi_##TYPENAME##_##OP##_reduce_block(dest, source, nreduce, start, stride,     \
                                                  size, pWrk, pSync);                            \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(RDXN_ON_STREAM_KERNEL)
#undef RDXN_ON_STREAM_KERNEL

#define CALL_RDXN_ON_STREAM(TYPENAME, TYPE, OP)                                                       \
    extern "C" void call_rdxn_##TYPENAME##_##OP##_on_stream_kern(                                     \
        TYPE *dest, const TYPE *source, size_t nreduce, int start, int stride, int size,              \
        TYPE *pWrk, long *pSync, cudaStream_t stream) {                                               \
        size_t num_threads_per_block =                                                                \
            (MAX_THREADS_PER_CTA > nreduce) ? nreduce : MAX_THREADS_PER_CTA;                          \
        int num_blocks = 1;                                                                           \
        rdxn_##TYPENAME##_##OP##_on_stream_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(  \
            dest, source, nreduce, start, stride, size, (uint64_t *)pWrk, (uint64_t *)pSync);         \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                                       \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(CALL_RDXN_ON_STREAM)
#undef CALL_RDXN_ON_STREAM
