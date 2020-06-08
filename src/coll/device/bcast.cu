/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"

#ifdef __CUDA_ARCH__

#define NVSHMEMI_GPU_BCAST_PUT2ALL(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, \
                                   PE_size, pSync)                                                \
    do {                                                                                          \
        int i;                                                                                    \
        int stride = 1 << logPE_stride;                                                           \
        int PE_end = PE_start + (stride * PE_size);                                               \
        if (PE_root == nvshmemi_mype_d) {                                                          \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                                 \
                nvshmem_put##SUFFIX##_nbi(dest, source, nelems, i);                               \
            }                                                                                     \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                          \
                nvshmem_put##SUFFIX##_nbi(dest, source, nelems, i);                               \
            }                                                                                     \
        }                                                                                         \
        nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                                  \
    } while (0)

#define NVSHMEMI_GET_ACTUAL_RANK(rank, effective_root, PE_start, PE_size, stride)       \
    do {                                                                                \
        rank = ((rank + effective_root) < PE_size) ? (rank + effective_root)            \
                                                   : (rank + effective_root - PE_size); \
        rank *= stride;                                                                 \
        rank += PE_start;                                                               \
    } while (0)

#define NVSHMEMI_GPU_BCAST_BINARY(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, \
                                  PE_size, pSync)                                                \
    do {                                                                                         \
        int offset;                                                                              \
        int i;                                                                                   \
        int stride = 1 << logPE_stride;                                                          \
        int PE_end = PE_start + (stride * (PE_size - 1));                                        \
        int effective_rank = (nvshmemi_mype_d - PE_start) / stride;                               \
        int effective_root = (PE_root - PE_start) / stride;                                      \
        int bin_tree_rank = -1;                                                                  \
        int left_child = -1;                                                                     \
        int right_child = -1;                                                                    \
        int parent = -1;                                                                         \
        bin_tree_rank = ((effective_rank - effective_root) >= 0)                                 \
                            ? (effective_rank - effective_root)                                  \
                            : (effective_rank - effective_root + PE_size);                       \
        left_child = (bin_tree_rank * 2) + 1;                                                    \
        right_child = (bin_tree_rank * 2) + 2;                                                   \
        parent = ((bin_tree_rank / 2) != ((bin_tree_rank - 1) / 2)) ? ((bin_tree_rank / 2) - 1)  \
                                                                    : (bin_tree_rank / 2);       \
        if (bin_tree_rank != 0) {                                                                \
            NVSHMEMI_GET_ACTUAL_RANK(parent, effective_root, PE_start, PE_size, stride);         \
            while (*((volatile long *)pSync) == NVSHMEM_SYNC_VALUE)                              \
                ;                                                                                \
            *((long *)pSync) = NVSHMEM_SYNC_VALUE;                                               \
            offset = ((bin_tree_rank / 2) != ((bin_tree_rank - 1) / 2)) ? 1 : 0;                 \
            nvshmem_long_signal((long *)pSync + offset, !NVSHMEM_SYNC_VALUE, parent);                 \
        }                                                                                        \
        if (left_child < PE_size) {                                                              \
            NVSHMEMI_GET_ACTUAL_RANK(left_child, effective_root, PE_start, PE_size, stride);     \
            nvshmem_put##SUFFIX(dest, source, nelems, left_child);                               \
            nvshmem_fence();                                                                     \
            nvshmemx_long_signal(pSync, !NVSHMEM_SYNC_VALUE, left_child);                              \
            left_child = 1;                                                                      \
        }                                                                                        \
        if (right_child < PE_size) {                                                             \
            NVSHMEMI_GET_ACTUAL_RANK(right_child, effective_root, PE_start, PE_size, stride);    \
            nvshmem_put##SUFFIX(dest, source, nelems, right_child);                              \
            nvshmem_fence();                                                                     \
            __threadfence();                                                                     \
            nvshmemx_long_signal(pSync, !NVSHMEM_SYNC_VALUE, right_child);                             \
            right_child = 1;                                                                     \
        }                                                                                        \
        if (left_child == 1) {                                                                   \
            while (*((volatile long *)pSync) == NVSHMEM_SYNC_VALUE)                              \
                ;                                                                                \
            *((long *)pSync) = NVSHMEM_SYNC_VALUE;                                               \
        }                                                                                        \
        if (right_child == 1) {                                                                  \
            while (*((volatile long *)pSync + 1) == NVSHMEM_SYNC_VALUE)                          \
                ;                                                                                \
            *((long *)pSync + 1) = NVSHMEM_SYNC_VALUE;                                           \
        }                                                                                        \
    } while (0)

#define NVSHMEMI_GPU_BCAST_PUT2ALL_DIRECT(SUFFIX, dest, source, nelems, PE_root, PE_start,       \
                                          logPE_stride, PE_size, pSync)                          \
    do {                                                                                         \
        int offset;                                                                              \
        char *round_dest;                                                                        \
        int i;                                                                                   \
        int stride = 1 << logPE_stride;                                                          \
        int PE_end = PE_start + (stride * PE_size);                                              \
        offset =                                                                                 \
            (char *)dest - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                          nvshmemi_mype_d));                                      \
        if (PE_root == nvshmemi_mype_d) {                                                         \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                                \
                round_dest =                                                                     \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +  \
                    offset;                                                                      \
                GPU_BITS_COPY_DIRECT(SUFFIX, round_dest, source, nelems);                        \
            }                                                                                    \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                         \
                round_dest =                                                                     \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +  \
                    offset;                                                                      \
                GPU_BITS_COPY_DIRECT(SUFFIX, round_dest, source, nelems);                        \
            }                                                                                    \
        }                                                                                        \
        nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                                 \
    } while (0)

#define NVSHMEMI_GPU_BCAST_PUT2ALL_POLL(SUFFIX, dest, source, nelems, PE_root, PE_start, \
                                        logPE_stride, PE_size, pSync)                    \
    do {                                                                                 \
        int i;                                                                           \
        int stride = 1 << logPE_stride;                                                  \
        int PE_end = PE_start + (stride * PE_size);                                      \
        if (PE_root == nvshmemi_mype_d) {                                                 \
            int j;                                                                       \
            uint32_t tmp[2];                                                             \
            uint32_t payld;                                                              \
            int subelems = (SUFFIX / 32);                                                \
            tmp[1] = 1;                                                                  \
            for (j = 0; j < nelems * subelems; j++) {                                    \
                payld = *((uint32_t *)source + j);                                       \
                tmp[0] = payld;                                                          \
                for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                    \
                    nvshmemx_long_signal((long *)pSync + j, *(long *)tmp, i);                  \
                }                                                                        \
                for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {             \
                    nvshmemx_long_signal((long *)pSync + j, *(long *)tmp, i);                  \
                }                                                                        \
            }                                                                            \
        } else {                                                                         \
            GPU_BITS_REG_CHECK(SUFFIX, dest, pSync, nelems);                             \
        }                                                                                \
        __threadfence();                                                                 \
    } while (0)

#define NVSHMEMI_GPU_BCAST_PUT2ALL_POLL_DIRECT(SUFFIX, dest, source, nelems, PE_root, PE_start,   \
                                               logPE_stride, PE_size, pSync)                      \
    do {                                                                                          \
        int offset;                                                                               \
        char *round_psync_dest;                                                                   \
        int i;                                                                                    \
        int stride = 1 << logPE_stride;                                                           \
        int PE_end = PE_start + (stride * PE_size);                                               \
        offset =                                                                                  \
            (char *)pSync - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                           nvshmemi_mype_d));                                      \
        if (PE_root == nvshmemi_mype_d) {                                                          \
            int j;                                                                                \
            uint32_t tmp[2];                                                                      \
            uint32_t payld;                                                                       \
            int subelems = (SUFFIX / 32);                                                         \
            tmp[1] = 1;                                                                           \
            for (j = 0; j < nelems * subelems; j++) {                                             \
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
            GPU_BITS_REG_CHECK(SUFFIX, dest, pSync, nelems);                                      \
        }                                                                                         \
        __threadfence();                                                                          \
    } while (0)

#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMI_GPU_BCAST(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, \
                           pSync)                                                                  \
    do {                                                                                           \
        NVSHMEMI_GPU_BCAST_PUT2ALL_DIRECT(SUFFIX, dest, source, nelems, PE_root, PE_start,         \
                                          logPE_stride, PE_size, pSync);                           \
    } while (0)
#else
#define NVSHMEMI_GPU_BCAST(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, \
                           pSync)                                                                  \
    do {                                                                                           \
        int psync_req_sz;                                                                          \
        int subelems = (sizeof(uint##SUFFIX##_t) / sizeof(uint32_t));                              \
        psync_req_sz = subelems * nelems;                                                          \
        if ((subelems > 0) && (psync_req_sz <= NVSHMEM_BCAST_SYNC_SIZE)) {                         \
            NVSHMEMI_GPU_BCAST_PUT2ALL_POLL_DIRECT(SUFFIX, dest, source, nelems, PE_root,          \
                                                   PE_start, logPE_stride, PE_size, pSync);        \
        } else {                                                                                   \
            NVSHMEMI_GPU_BCAST_PUT2ALL_DIRECT(SUFFIX, dest, source, nelems, PE_root, PE_start,     \
                                              logPE_stride, PE_size, pSync);                       \
        }                                                                                          \
    } while (0)
#endif
#else
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMI_GPU_BCAST(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, \
                           pSync)                                                                  \
    do {                                                                                           \
        NVSHMEMI_GPU_BCAST_PUT2ALL(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride,  \
                                   PE_size, pSync);                                                \
    } while (0)
#else
#define NVSHMEMI_GPU_BCAST(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, \
                           pSync)                                                                  \
    do {                                                                                           \
        int psync_req_sz;                                                                          \
        int subelems = (sizeof(uint##SUFFIX##_t) / sizeof(uint32_t));                              \
        psync_req_sz = subelems * nelems;                                                          \
        if ((subelems > 0) && (psync_req_sz <= NVSHMEM_BCAST_SYNC_SIZE)) {                         \
            NVSHMEMI_GPU_BCAST_PUT2ALL_POLL(SUFFIX, dest, source, nelems, PE_root, PE_start,       \
                                            logPE_stride, PE_size, pSync);                         \
        } else {                                                                                   \
            NVSHMEMI_GPU_BCAST_PUT2ALL(SUFFIX, dest, source, nelems, PE_root, PE_start,            \
                                       logPE_stride, PE_size, pSync);                              \
        }                                                                                          \
    } while (0)
#endif
#endif

#define DEFN_NVSHMEM_GPU_BCAST(SUFFIX)                                                             \
    __device__ void nvshmem_broadcast##SUFFIX(void *dest, const void *source, size_t nelems,       \
                                              int PE_root, int PE_start, int logPE_stride,         \
                                              int PE_size, long *pSync) {                          \
        NVSHMEMI_GPU_BCAST(SUFFIX, dest, source, nelems, PE_root, PE_start, logPE_stride, PE_size, \
                           pSync);                                                                 \
    }

#define DEFN_NVSHMEM_GPU_BCAST_TYPES() \
    DEFN_NVSHMEM_GPU_BCAST(32);        \
    DEFN_NVSHMEM_GPU_BCAST(64);

DEFN_NVSHMEM_GPU_BCAST_TYPES();

#endif
