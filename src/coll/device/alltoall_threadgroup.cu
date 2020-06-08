/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include <cstdio>
#include <cassert>

#ifdef __CUDA_ARCH__

#define NVSHMEMI_GPU_ALLTOALL_THREADGROUP_P2P_ALLPUSH(SC, SUFFIX, dest, source, nelems, PE_start, \
                                                      logPE_stride, PE_size, pSync)               \
    do {                                                                                          \
        int stride = 1 << logPE_stride;                                                           \
        int next_rank;                                                                            \
        int src_offset;                                                                           \
        int dst_offset;                                                                           \
        int mype = nvshmemi_mype_d;                                                                \
                                                                                                  \
        for (int ii = 0; ii < PE_size; ii++) {                                                    \
            next_rank = (mype + (ii * stride)) % (stride * PE_size);                              \
            src_offset = nelems * ((next_rank - PE_start) / stride);                              \
            dst_offset = nelems * ((mype - PE_start) / stride);                                   \
            nvshmemx_put##SUFFIX##_nbi_##SC((uint##SUFFIX##_t *)dest + dst_offset,                \
                                            (uint##SUFFIX##_t *)source + src_offset, nelems,      \
                                            next_rank);                                           \
        }                                                                                         \
        nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                            \
    } while (0)

#define NVSHMEMI_GPU_ALLTOALL_THREADGROUP_ALLPUSH(SC, SUFFIX, dest, source, nelems, PE_start,    \
                                                  logPE_stride, PE_size, pSync)                  \
    do {                                                                                         \
        int stride = 1 << logPE_stride;                                                          \
        int next_rank;                                                                           \
        int src_offset;                                                                          \
        int dst_offset;                                                                          \
        int mype = nvshmemi_mype_d;                                                               \
        int offset;                                                                              \
        char *round_dest;                                                                        \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                         \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                   \
        offset =                                                                                 \
            (char *)dest - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                          nvshmemi_mype_d));                                      \
                                                                                                 \
        for (int ii = 0; ii < PE_size; ii++) {                                                   \
            next_rank = (mype + (ii * stride)) % (stride * PE_size);                             \
            src_offset = nelems * ((next_rank - PE_start) / stride);                             \
            dst_offset = nelems * ((mype - PE_start) / stride);                                  \
            round_dest = (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +   \
                                        next_rank)) +                                            \
                         offset + sizeof(uint##SUFFIX##_t) * dst_offset;                         \
            GPU_BITS_COPY_THREADGROUP_DIRECT(SUFFIX, (uint##SUFFIX##_t *)round_dest,             \
                                             (uint##SUFFIX##_t *)source + src_offset, nelems,    \
                                             myIdx, groupSize);                                  \
        }                                                                                        \
        nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                           \
    } while (0)

#ifdef NVSHMEM_GPU_COLL_USE_LDST
#define NVSHMEMI_GPU_ALLTOALL_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_start,         \
                                          logPE_stride, PE_size, pSync)                       \
    do {                                                                                      \
        NVSHMEMI_GPU_ALLTOALL_THREADGROUP_ALLPUSH(SC, SUFFIX, dest, source, nelems, PE_start, \
                                                  logPE_stride, PE_size, pSync);              \
    } while (0)
#else
#define NVSHMEMI_GPU_ALLTOALL_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_start,             \
                                          logPE_stride, PE_size, pSync)                           \
    do {                                                                                          \
        NVSHMEMI_GPU_ALLTOALL_THREADGROUP_P2P_ALLPUSH(SC, SUFFIX, dest, source, nelems, PE_start, \
                                                      logPE_stride, PE_size, pSync);              \
    } while (0)
#endif

#define DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP(SC, SUFFIX)                                         \
    __device__ void nvshmemx_alltoall##SUFFIX##_##SC(void *dest, const void *source,               \
                                                     size_t nelems, int PE_start,                  \
                                                     int logPE_stride, int PE_size, long *pSync) { \
        NVSHMEMI_GPU_ALLTOALL_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_start,              \
                                          logPE_stride, PE_size, pSync);                           \
    }

#define DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP_TYPES(SC) \
    DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP(SC, 8);       \
    DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP(SC, 16);      \
    DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP(SC, 32);      \
    DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP(SC, 64);

DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP_TYPES(warp);
DEFN_NVSHMEMX_GPU_ALLTOALL_THREADGROUP_TYPES(block);

#endif

#define ALLTOALL_ON_STREAM_KERNEL(BITS)                                                            \
    __global__ void alltoall##BITS##_on_stream_kernel(                                             \
        void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,             \
        int PE_size, long *pSync) {                                                                \
        if (!blockIdx.x)                                                                           \
            nvshmemx_alltoall##BITS##_block(dest, source, nelems, PE_start, logPE_stride, PE_size, \
                                            pSync);                                                \
    }

ALLTOALL_ON_STREAM_KERNEL(32);
ALLTOALL_ON_STREAM_KERNEL(64);

#define CALL_ALLTOALL_ON_STREAM(BITS)                                                              \
    extern "C" void call_alltoall##BITS##_on_stream_kern(                                          \
        void *dest, const void *source, size_t nelems, int PE_start, int logPE_stride,             \
        int PE_size, long *pSync, cudaStream_t stream) {                                           \
        int num_threads_per_block = (MAX_THREADS_PER_CTA > nelems) ? nelems : MAX_THREADS_PER_CTA; \
        int num_blocks = 1;                                                                        \
        alltoall##BITS##_on_stream_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(       \
            dest, source, nelems, PE_start, logPE_stride, PE_size, pSync);                         \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                                    \
    }

CALL_ALLTOALL_ON_STREAM(32);
CALL_ALLTOALL_ON_STREAM(64);
