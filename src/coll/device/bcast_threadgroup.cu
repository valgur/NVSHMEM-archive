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

#define NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL(SC, SUFFIX, dest, source, nelems, PE_root, \
                                               PE_start, logPE_stride, PE_size, pSync)    \
    do {                                                                                  \
        int i;                                                                            \
        int stride = 1 << logPE_stride;                                                   \
        int PE_end = PE_start + (stride * PE_size);                                       \
        if (PE_root == nvshmemi_mype_d) {                                                  \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                         \
                nvshmemx_put##SUFFIX##_nbi_##SC(dest, source, nelems, i);                 \
            }                                                                             \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                  \
                nvshmemx_put##SUFFIX##_nbi_##SC(dest, source, nelems, i);                 \
            }                                                                             \
        }                                                                                 \
        nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                    \
    } while (0)

#define NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_DIRECT(SC, SUFFIX, dest, source, nelems, PE_root, \
                                                      PE_start, logPE_stride, PE_size, pSync)    \
    do {                                                                                         \
        int offset;                                                                              \
        char *round_dest;                                                                        \
        int i;                                                                                   \
        int stride = 1 << logPE_stride;                                                          \
        int PE_end = PE_start + (stride * PE_size);                                              \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                         \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                   \
        offset =                                                                                 \
            (char *)dest - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                          nvshmemi_mype_d));                                      \
        if (PE_root == nvshmemi_mype_d) {                                                         \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                                \
                round_dest =                                                                     \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +  \
                    offset;                                                                      \
                GPU_BITS_COPY_THREADGROUP_DIRECT(SUFFIX, round_dest, source, nelems, myIdx,      \
                                                 groupSize);                                     \
            }                                                                                    \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                         \
                round_dest =                                                                     \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +  \
                    offset;                                                                      \
                GPU_BITS_COPY_THREADGROUP_DIRECT(SUFFIX, round_dest, source, nelems, myIdx,      \
                                                 groupSize);                                     \
            }                                                                                    \
        }                                                                                        \
        nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                           \
    } while (0)

#define NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_POLL(SC, SUFFIX, dest, source, nelems, PE_root, \
                                                    PE_start, logPE_stride, PE_size, pSync)    \
    do {                                                                                       \
        int i;                                                                                 \
        int stride = 1 << logPE_stride;                                                        \
        int PE_end = PE_start + (stride * PE_size);                                            \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                       \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                 \
        if (PE_root == nvshmemi_mype_d) {                                                       \
            int j;                                                                             \
            uint32_t tmp[2];                                                                   \
            uint32_t payld;                                                                    \
            int subelems = (SUFFIX / 32);                                                      \
            tmp[1] = 1;                                                                        \
            for (j = myIdx; j < nelems * subelems; j += groupSize) {                           \
                payld = *((uint32_t *)source + j);                                             \
                tmp[0] = payld;                                                                \
                for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                          \
                    nvshmemx_long_signal(((long *)pSync + j), *((long *)tmp), i);                    \
                }                                                                              \
                for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                   \
                    nvshmemx_long_signal(((long *)pSync + j), *((long *)tmp), i);                    \
                }                                                                              \
            }                                                                                  \
        } else {                                                                               \
            GPU_BITS_REG_CHECK_THREADGROUP(SUFFIX, dest, pSync, nelems, myIdx, groupSize);     \
        }                                                                                      \
        __threadfence();                                                                       \
    } while (0)

#define NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_POLL_DIRECT(                                       \
    SC, SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, pSync)            \
    do {                                                                                          \
        int offset;                                                                               \
        char *round_psync_dest;                                                                   \
        int i;                                                                                    \
        int stride = 1 << logPE_stride;                                                           \
        int PE_end = PE_start + (stride * PE_size);                                               \
        offset =                                                                                  \
            (char *)pSync - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                           nvshmemi_mype_d));                                      \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                          \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                    \
        if (PE_root == nvshmemi_mype_d) {                                                          \
            int j;                                                                                \
            uint32_t tmp[2];                                                                      \
            uint32_t payld;                                                                       \
            int subelems = (SUFFIX / 32);                                                         \
            tmp[1] = 1;                                                                           \
            for (j = myIdx; j < nelems * subelems; j += groupSize) {                              \
                payld = *((uint32_t *)source + j);                                                \
                tmp[0] = payld;                                                                   \
                for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                             \
                    round_psync_dest =                                                            \
                        (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +     \
                                       i)) +                                                      \
                        offset;                                                                   \
                    *((uint64_t *)round_psync_dest + j) = *((uint64_t *)tmp);                     \
                }                                                                                 \
                for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                      \
                    round_psync_dest =                                                            \
                        (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +     \
                                       i)) +                                                      \
                        offset;                                                                   \
                    *((uint64_t *)round_psync_dest + j) = *((uint64_t *)tmp);                     \
                }                                                                                 \
            }                                                                                     \
        } else {                                                                                  \
            GPU_BITS_REG_CHECK_THREADGROUP(SUFFIX, dest, pSync, nelems, myIdx, groupSize);        \
        }                                                                                         \
        __threadfence();                                                                          \
    } while (0)

#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMI_GPU_BCAST_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_root, PE_start,      \
                                       logPE_stride, PE_size, pSync)                             \
    do {                                                                                         \
        NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_DIRECT(SC, SUFFIX, dest, source, nelems, PE_root, \
                                                      PE_start, logPE_stride, PE_size, pSync);   \
        NVSHMEMI_SYNC_##SC();                                                                    \
    } while (0)
#else
#define NVSHMEMI_GPU_BCAST_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_root, PE_start,      \
                                       logPE_stride, PE_size, pSync)                             \
    do {                                                                                         \
        int psync_req_sz;                                                                        \
        int subelems = (sizeof(uint##SUFFIX##_t) / sizeof(uint32_t));                            \
        psync_req_sz = subelems * nelems;                                                        \
        if ((subelems > 0) && (psync_req_sz <= NVSHMEM_BCAST_SYNC_SIZE)) {                       \
            NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_POLL_DIRECT(SC, SUFFIX, dest, source, nelems, \
                                                               PE_root, PE_start, logPE_stride,  \
                                                               PE_size, pSync);                  \
        } else {                                                                                 \
            NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_DIRECT(SC, SUFFIX, dest, source, nelems,      \
                                                          PE_root, PE_start, logPE_stride,       \
                                                          PE_size, pSync);                       \
        }                                                                                        \
        NVSHMEMI_SYNC_##SC();                                                                    \
    } while (0)
#endif
#else
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMI_GPU_BCAST_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_root, PE_start, \
                                       logPE_stride, PE_size, pSync)                        \
    do {                                                                                    \
        NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL(SC, SUFFIX, dest, source, nelems, PE_root,   \
                                               PE_start, logPE_stride, PE_size, pSync);     \
        NVSHMEMI_SYNC_##SC();                                                               \
    } while (0)
#else
#define NVSHMEMI_GPU_BCAST_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_root, PE_start,        \
                                       logPE_stride, PE_size, pSync)                               \
    do {                                                                                           \
        int psync_req_sz;                                                                          \
        int subelems = (sizeof(uint##SUFFIX##_t) / sizeof(uint32_t));                              \
        psync_req_sz = subelems * nelems;                                                          \
        if ((subelems > 0) && (psync_req_sz <= NVSHMEM_BCAST_SYNC_SIZE)) {                         \
            NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_POLL(SC, SUFFIX, dest, source, nelems, PE_root, \
                                                        PE_start, logPE_stride, PE_size, pSync);   \
        } else {                                                                                   \
            NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL(SC, SUFFIX, dest, source, nelems, PE_root,      \
                                                   PE_start, logPE_stride, PE_size, pSync);        \
        }                                                                                          \
        NVSHMEMI_SYNC_##SC();                                                                      \
    } while (0)
#endif
#endif

#define DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP(SC, SUFFIX)                                     \
    __device__ void nvshmemx_broadcast##SUFFIX##_##SC(                                      \
        void *dest, const void *source, size_t nelems, int PE_root, int PE_start,           \
        int logPE_stride, int PE_size, long *pSync) {                                       \
        NVSHMEMI_GPU_BCAST_THREADGROUP(SC, SUFFIX, dest, source, nelems, PE_root, PE_start, \
                                       logPE_stride, PE_size, pSync);                       \
    }

#define DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP_TYPES(SC) \
    DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP(SC, 8);       \
    DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP(SC, 16);      \
    DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP(SC, 32);      \
    DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP(SC, 64);

DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP_TYPES(warp);
DEFN_NVSHMEMX_GPU_BCAST_THREADGROUP_TYPES(block);

#endif

#define BCAST_ON_STREAM_KERNEL(BITS)                                                  \
    __global__ void broadcast##BITS##_on_stream_kernel(                               \
        void *dest, const void *source, size_t nelems, int PE_root, int PE_start,     \
        int logPE_stride, int PE_size, long *pSync) {                                 \
        if (!blockIdx.x)                                                              \
            nvshmemx_broadcast##BITS##_block(dest, source, nelems, PE_root, PE_start, \
                                             logPE_stride, PE_size, pSync);           \
    }

BCAST_ON_STREAM_KERNEL(32);
BCAST_ON_STREAM_KERNEL(64);

#define CALL_BCAST_ON_STREAM(BITS)                                                                 \
    extern "C" void call_broadcast##BITS##_on_stream_kern(                                         \
        void *dest, const void *source, size_t nelems, int PE_root, int PE_start,                  \
        int logPE_stride, int PE_size, long *pSync, cudaStream_t stream) {                         \
        int num_threads_per_block = (MAX_THREADS_PER_CTA > nelems) ? nelems : MAX_THREADS_PER_CTA; \
        int num_blocks = 1;                                                                        \
        broadcast##BITS##_on_stream_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(      \
            dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, pSync);                \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                                    \
    }

CALL_BCAST_ON_STREAM(32);
CALL_BCAST_ON_STREAM(64);
