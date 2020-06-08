/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "gpu_coll.h"

#ifdef __CUDA_ARCH__

#define NVSHMEMI_GPU_COLLECT_P2P_ALL_PUSH(SUFFIX, dest, source, nelems, PE_start, logPE_stride, \
                                          PE_size, pSync)                                       \
    do {                                                                                        \
        int stride = 1 << logPE_stride;                                                         \
        int next_rank;                                                                          \
        int next_offset;                                                                        \
        int mype = nvshmemi_mype_d;                                                              \
                                                                                                \
        for (int ii = 0; ii < PE_size; ii++) {                                                  \
            next_rank = (mype + (ii * stride)) % (stride * PE_size);                            \
            next_offset = nelems * ((mype - PE_start) / stride);                                \
            /*XXX:typecast dest to suppress warning "arithmetic on pointer to void or function  \
             * typ"*/                                                                           \
            nvshmem_put##SUFFIX##_nbi((uint##SUFFIX##_t *)dest + next_offset, source, nelems,   \
                                      next_rank);                                               \
        }                                                                                       \
        nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                                \
    } while (0)

#define NVSHMEMI_GPU_COLLECT_ALL_PUSH(SUFFIX, dest, source, nelems, PE_start, logPE_stride,      \
                                      PE_size, pSync)                                            \
    do {                                                                                         \
        int stride = 1 << logPE_stride;                                                          \
        int next_rank;                                                                           \
        int next_offset;                                                                         \
        int mype = nvshmemi_mype_d;                                                               \
        int offset;                                                                              \
        char *round_dest;                                                                        \
        offset =                                                                                 \
            (char *)dest - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                          nvshmemi_mype_d));                                      \
                                                                                                 \
        for (int ii = 0; ii < PE_size; ii++) {                                                   \
            next_rank = (mype + (ii * stride)) % (stride * PE_size);                             \
            next_offset = nelems * ((mype - PE_start) / stride);                                 \
            round_dest = (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +   \
                                        next_rank)) +                                            \
                         offset + sizeof(uint##SUFFIX##_t) * next_offset;                        \
            GPU_BITS_COPY_DIRECT(SUFFIX, (uint##SUFFIX##_t *)round_dest, source, nelems);        \
        }                                                                                        \
        nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                                 \
    } while (0)

#ifdef NVSHMEM_GPU_COLL_USE_LDST
#define NVSHMEMI_GPU_COLLECT(SUFFIX, dest, source, nelems, PE_start, logPE_stride, PE_size, pSync) \
    do {                                                                                           \
        NVSHMEMI_GPU_COLLECT_ALL_PUSH(SUFFIX, dest, source, nelems, PE_start, logPE_stride,        \
                                      PE_size, pSync);                                             \
    } while (0)
#else
#define NVSHMEMI_GPU_COLLECT(SUFFIX, dest, source, nelems, PE_start, logPE_stride, PE_size, pSync) \
    do {                                                                                           \
        NVSHMEMI_GPU_COLLECT_P2P_ALL_PUSH(SUFFIX, dest, source, nelems, PE_start, logPE_stride,    \
                                          PE_size, pSync);                                         \
    } while (0)
#endif

#define DEFN_NVSHMEM_GPU_COLLECT(SUFFIX)                                                    \
    __device__ void nvshmem_collect##SUFFIX(void *dest, const void *source, size_t nelems,  \
                                            int PE_start, int logPE_stride, int PE_size,    \
                                            long *pSync) {                                  \
        NVSHMEMI_GPU_COLLECT(SUFFIX, dest, source, nelems, PE_start, logPE_stride, PE_size, \
                             pSync);                                                        \
    }

#define DEFN_NVSHMEM_GPU_COLLECT_TYPES() \
    DEFN_NVSHMEM_GPU_COLLECT(32);        \
    DEFN_NVSHMEM_GPU_COLLECT(64);

DEFN_NVSHMEM_GPU_COLLECT_TYPES();

#endif
