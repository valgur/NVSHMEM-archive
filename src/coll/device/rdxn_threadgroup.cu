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

#define GPU_HEAD_CHECK_OP_THREADGROUP(TYPE, OP, dest, src, actual_src, nelems, myIdx, groupSize) \
    do {                                                                                         \
        int i, k;                                                                                \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                          \
        volatile uint32_t *header = NULL;                                                        \
        TYPE tmp;                                                                                \
        uint32_t *tmp_ptr = (uint32_t *)&tmp;                                                    \
        uint32_t *payload = NULL;                                                                \
        for (i = myIdx; i < nelems; i += groupSize) {                                            \
            for (k = 0; k < subelems; k++) {                                                     \
                payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k);                    \
                header = (uint32_t *)payload + 1;                                                \
                while (1 != *header)                                                             \
                    ;                                                                            \
                *header = 0;                                                                     \
                *(tmp_ptr + k) = *payload;                                                       \
            }                                                                                    \
            perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)actual_src + i), tmp);            \
        }                                                                                        \
    } while (0)

#define GPU_HEAD_CHECKALL_OP_THREADGROUP(TYPE, OP, dest, src, actual_src, nelems, PE_start, \
                                         stride, PE_size, myIdx, groupSize)                 \
    do {                                                                                    \
        int i, j, k;                                                                        \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                     \
        volatile uint32_t *header = NULL;                                                   \
        TYPE tmp;                                                                           \
        uint32_t *tmp_ptr = (uint32_t *)&tmp;                                               \
        uint32_t *payload = NULL;                                                           \
        int my_active_set_pe = ((nvshmemi_mype_d - PE_start) / stride);                      \
        TYPE *src_ptr = (TYPE *)actual_src;                                                 \
                                                                                            \
        for (j = (my_active_set_pe - 1); j >= 0; j--) {                                     \
            for (i = myIdx; i < nelems; i += groupSize) {                                   \
                for (k = 0; k < subelems; k++) {                                            \
                    payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k +           \
                                           (nelems * subelems * j));                        \
                    header = (uint32_t *)payload + 1;                                       \
                    while (1 != *header)                                                    \
                        ;                                                                   \
                    *header = 0;                                                            \
                    *(tmp_ptr + k) = *payload;                                              \
                }                                                                           \
                perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);      \
            }                                                                               \
            src_ptr = dest;                                                                 \
        }                                                                                   \
        for (j = PE_size - 1; j > my_active_set_pe; j--) {                                  \
            for (i = myIdx; i < nelems; i += groupSize) {                                   \
                for (k = 0; k < subelems; k++) {                                            \
                    payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k +           \
                                           (nelems * subelems * j));                        \
                    header = (uint32_t *)payload + 1;                                       \
                    while (1 != *header)                                                    \
                        ;                                                                   \
                    *header = 0;                                                            \
                    *(tmp_ptr + k) = *payload;                                              \
                }                                                                           \
                perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);      \
            }                                                                               \
            src_ptr = dest;                                                                 \
        }                                                                                   \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP(TYPE, OP, x, y, z, nelems, myIdx, groupSize)     \
    do {                                                                               \
        int i;                                                                         \
        for (i = myIdx; i < nelems; i += groupSize) {                                  \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i)); \
        }                                                                              \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP_P2P_GET(SC, TYPE, OP, x, y, next_rank, z, nelems, myIdx,   \
                                              groupSize, PE_start, logPE_stride, PE_size, pSync) \
    do {                                                                                         \
        int i;                                                                                   \
        int group_nelems = ((nelems / groupSize) * groupSize);                                   \
        int excess = nelems - group_nelems;                                                      \
        for (i = myIdx; i < group_nelems; i += groupSize) {                                      \
            nvshmem_##TYPE##_get((TYPE *)((TYPE *)gpu_ipwrk_d + myIdx), (TYPE *)((TYPE *)y + i), \
                                 1, next_rank);                                                  \
            nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                       \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                              \
                                *((TYPE *)gpu_ipwrk_d + myIdx));                                 \
        }                                                                                        \
        if (excess) {                                                                            \
            if (i < nelems) {                                                                    \
                nvshmem_##TYPE##_get((TYPE *)((TYPE *)gpu_ipwrk_d + myIdx),                      \
                                     (TYPE *)((TYPE *)y + i), 1, next_rank);                     \
            }                                                                                    \
            nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                       \
            if (i < nelems) {                                                                    \
                perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                          \
                                    *((TYPE *)gpu_ipwrk_d + myIdx));                             \
            }                                                                                    \
        }                                                                                        \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT(SC, TYPE, OP, x, y, next_rank, z, offset, nelems,  \
                                              myIdx, groupSize, PE_start, logPE_stride, PE_size, \
                                              pSync)                                             \
    do {                                                                                         \
        int i;                                                                                   \
        int group_nelems = ((nelems / groupSize) * groupSize);                                   \
        int excess = nelems - group_nelems;                                                      \
        for (i = myIdx; i < group_nelems; i += groupSize) {                                      \
            nvshmem_##TYPE##_put_nbi(                                                            \
                (TYPE *)((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize))),            \
                (TYPE *)((TYPE *)y + i), 1, next_rank);                                          \
            nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                       \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                              \
                                *((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize))));  \
        }                                                                                        \
        offset++;                                                                                \
        if (excess) {                                                                            \
            if (i < nelems) {                                                                    \
                nvshmem_##TYPE##_put_nbi(                                                        \
                    (TYPE *)((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize))),        \
                    (TYPE *)((TYPE *)y + i), 1, next_rank);                                      \
            }                                                                                    \
            nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                       \
            if (i < nelems) {                                                                    \
                perform_gpu_rd_##OP(                                                             \
                    *((TYPE *)z + i), *((TYPE *)x + i),                                          \
                    *((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize))));              \
            }                                                                                    \
        }                                                                                        \
        offset++;                                                                                \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT_DIRECT(SC, TYPE, OP, x, y, next_rank, z, offset,  \
                                                     nelems, myIdx, groupSize, PE_start,        \
                                                     logPE_stride, PE_size, pSync)              \
    do {                                                                                        \
        int i;                                                                                  \
        int group_nelems = ((nelems / groupSize) * groupSize);                                  \
        int excess = nelems - group_nelems;                                                     \
        for (i = myIdx; i < group_nelems; i += groupSize) {                                     \
            *((TYPE *)((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize)))) =           \
                *((TYPE *)((TYPE *)y + i));                                                     \
            nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                      \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                             \
                                *((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize)))); \
        }                                                                                       \
        offset++;                                                                               \
        if (excess) {                                                                           \
            if (i < nelems) {                                                                   \
                *((TYPE *)((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize)))) =       \
                    *((TYPE *)((TYPE *)y + i));                                                 \
            }                                                                                   \
            nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                      \
            if (i < nelems) {                                                                   \
                perform_gpu_rd_##OP(                                                            \
                    *((TYPE *)z + i), *((TYPE *)x + i),                                         \
                    *((TYPE *)gpu_ipwrk_d + (myIdx + ((offset & 1) * groupSize))));             \
            }                                                                                   \
        }                                                                                       \
        offset++;                                                                               \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_ZCOPY_GET_BAR_DIRECT(                                       \
    SC, TYPE, OP, dest, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync)             \
    do {                                                                                           \
        int stride = 1 << logPE_stride;                                                            \
        int next_rank = -1;                                                                        \
        int src_offset = -1;                                                                       \
        int next_offset = -1;                                                                      \
        char *base = NULL;                                                                         \
        char *peer_base = NULL;                                                                    \
        char *peer_source = NULL;                                                                  \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int i;                                                                                     \
                                                                                                   \
        base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +       \
                                      nvshmemi_mype_d));                                            \
        src_offset = ((char *)source - base);                                                      \
                                                                                                   \
        next_rank = (nvshmemi_mype_d + (stride)) % (stride * PE_size);                              \
        next_offset = src_offset;                                                                  \
        peer_base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +  \
                                           next_rank));                                            \
        peer_source = peer_base + next_offset;                                                     \
        GPU_LINEAR_REDUCE_THREADGROUP(TYPE, OP, (void *)source, peer_source, dest, nreduce, myIdx, \
                                      groupSize);                                                  \
                                                                                                   \
        for (i = 2; i < PE_size; i++) {                                                            \
            next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                      \
            next_offset = src_offset;                                                              \
            peer_base = (char *)((void *)__ldg(                                                    \
                (const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));                \
            peer_source = peer_base + next_offset;                                                 \
            GPU_LINEAR_REDUCE_THREADGROUP(TYPE, OP, dest, peer_source, dest, nreduce, myIdx,       \
                                          groupSize);                                              \
        }                                                                                          \
        nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                             \
    } while (0)

// barrier limited
#define NVSHMEMXI_GPU_RDXN_THREADGROUP_GET_BAR(SC, TYPE, OP, dest, source, nreduce, PE_start,    \
                                               logPE_stride, PE_size, pWrk, pSync)               \
    do {                                                                                         \
        int stride = 1 << logPE_stride;                                                          \
        int next_rank = -1;                                                                      \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                         \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                   \
        int i;                                                                                   \
                                                                                                 \
        next_rank = (nvshmemi_mype_d + (stride)) % (stride * PE_size);                            \
        GPU_LINEAR_REDUCE_THREADGROUP_P2P_GET(SC, TYPE, OP, source, source, next_rank, dest,     \
                                              nreduce, myIdx, groupSize, PE_start, logPE_stride, \
                                              PE_size, pSync);                                   \
                                                                                                 \
        for (i = 2; i < PE_size; i++) {                                                          \
            next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                    \
            GPU_LINEAR_REDUCE_THREADGROUP_P2P_GET(SC, TYPE, OP, dest, source, next_rank, dest,   \
                                                  nreduce, myIdx, groupSize, PE_start,           \
                                                  logPE_stride, PE_size, pSync);                 \
        }                                                                                        \
        NVSHMEMI_SYNC_##SC();                                                                    \
    } while (0)

// barrier limited
#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUT_BAR(SC, TYPE, OP, dest, source, nreduce, PE_start,   \
                                               logPE_stride, PE_size, pWrk, pSync)              \
    do {                                                                                        \
        int stride = 1 << logPE_stride;                                                         \
        int next_rank = -1;                                                                     \
        int counter = 0;                                                                        \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                        \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                  \
        int i;                                                                                  \
                                                                                                \
        next_rank = (nvshmemi_mype_d + (stride)) % (stride * PE_size);                           \
        GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT(SC, TYPE, OP, source, source, next_rank, dest,    \
                                              counter, nreduce, myIdx, groupSize, PE_start,     \
                                              logPE_stride, PE_size, pSync);                    \
                                                                                                \
        for (i = 2; i < PE_size; i++) {                                                         \
            next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                   \
            GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT(SC, TYPE, OP, dest, source, next_rank, dest,  \
                                                  counter, nreduce, myIdx, groupSize, PE_start, \
                                                  logPE_stride, PE_size, pSync);                \
        }                                                                                       \
        NVSHMEMI_SYNC_##SC();                                                                   \
    } while (0)

#define GPU_RDXN_ON_DEMAND_THREADGROUP(SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride, \
                                       PE_size, pWrk, pSync)                                       \
    do {                                                                                           \
        int stride = 1 << logPE_stride;                                                            \
        int next_rank = -1;                                                                        \
        TYPE *op1 = NULL, *op2 = NULL;                                                             \
        int i;                                                                                     \
        volatile TYPE *tmp_operand;                                                                \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
                                                                                                   \
        tmp_operand = (TYPE *)gpu_own_intm_rdxn_addr_d;                                            \
                                                                                                   \
        nvshmemx_##TYPE##_put_##SC((TYPE *)dest, (TYPE *)source, nelems, nvshmemi_mype_d);          \
        for (i = 1; i < PE_size; i++) {                                                            \
            next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                      \
            nvshmemx_##TYPE##_put_##SC((TYPE *)tmp_operand, (TYPE *)source, nelems, next_rank);    \
            nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                         \
            op1 = (TYPE *)dest;                                                                    \
            op2 = (TYPE *)tmp_operand;                                                             \
            GPU_LINEAR_REDUCE_THREADGROUP(TYPE, OP, op1, op2, op1, nelems, myIdx, groupSize);      \
            nvshmemx_sync_##SC(PE_start, logPE_stride, PE_size, pSync);                            \
        }                                                                                          \
        NVSHMEMI_SYNC_##SC();                                                                      \
    } while (0)

#define GPU_RDXN_SEGMENT_THREADGROUP(SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride,  \
                                     PE_size, pWrk, pSync)                                        \
    do {                                                                                          \
        int type_size = sizeof(TYPE);                                                             \
        int msg_len = nelems * type_size;                                                         \
        int stride = 1 << logPE_stride;                                                           \
        int next_rank = -1;                                                                       \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                          \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                    \
        TYPE *op1 = NULL, *op2 = NULL;                                                            \
        int i, j;                                                                                 \
        volatile TYPE *tmp_operand;                                                               \
        int remainder = 0;                                                                        \
        int rnds_floor = 0;                                                                       \
        int offset = 0;                                                                           \
        int pe_offset = 0;                                                                        \
        int pes_per_round = 0;                                                                    \
        int round = 0;                                                                            \
        int exchange_size = 0;                                                                    \
        int nvshm_gpu_rdxn_seg_size = gpu_coll_env_params_var_d.gpu_intm_rdxn_size;               \
                                                                                                  \
        tmp_operand = (TYPE *)gpu_own_intm_rdxn_addr_d;                                           \
        nvshmemx_##TYPE##_put_nbi_##SC((TYPE *)dest, (TYPE *)source, nelems, nvshmemi_mype_d);     \
                                                                                                  \
        rnds_floor = msg_len / nvshm_gpu_rdxn_seg_size;                                           \
        remainder = msg_len % nvshm_gpu_rdxn_seg_size;                                            \
                                                                                                  \
        for (j = 0; j < rnds_floor; j++) {                                                        \
            exchange_size = nvshm_gpu_rdxn_seg_size;                                              \
            for (i = 1; i < PE_size; i++) {                                                       \
                next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                 \
                nvshmemx_##TYPE##_put_nbi_##SC((TYPE *)tmp_operand, (TYPE *)source + offset,      \
                                               (exchange_size / sizeof(TYPE)), next_rank);        \
                nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                    \
                op1 = (TYPE *)dest + offset;                                                      \
                op2 = (TYPE *)tmp_operand;                                                        \
                GPU_LINEAR_REDUCE_THREADGROUP(TYPE, OP, op1, op2, op1,                            \
                                              (exchange_size / sizeof(TYPE)), myIdx, groupSize);  \
                nvshmemx_sync_##SC(PE_start, logPE_stride, PE_size, pSync);                       \
            }                                                                                     \
            offset += (exchange_size / sizeof(TYPE));                                             \
        }                                                                                         \
                                                                                                  \
        if (remainder != 0) {                                                                     \
            exchange_size = remainder;                                                            \
            pes_per_round = nvshm_gpu_rdxn_seg_size / remainder;                                  \
            pe_offset = 1;                                                                        \
            do {                                                                                  \
                round = 0;                                                                        \
                for (i = pe_offset; ((round < pes_per_round) && (i < PE_size)); i++) {            \
                    next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);             \
                    nvshmemx_##TYPE##_put_nbi_##SC(                                               \
                        (TYPE *)((TYPE *)tmp_operand + (round * (exchange_size / sizeof(TYPE)))), \
                        (TYPE *)source + offset, (exchange_size / sizeof(TYPE)), next_rank);      \
                    round++;                                                                      \
                    pe_offset++;                                                                  \
                }                                                                                 \
                nvshmemx_barrier_##SC(PE_start, logPE_stride, PE_size, pSync);                    \
                for (i = 0; i < round; i++) {                                                     \
                    op1 = (TYPE *)dest + offset;                                                  \
                    op2 = (TYPE *)((TYPE *)tmp_operand + (i * (exchange_size / sizeof(TYPE))));   \
                    GPU_LINEAR_REDUCE_THREADGROUP(TYPE, OP, op1, op2, op1,                        \
                                                  (exchange_size / sizeof(TYPE)), myIdx,          \
                                                  groupSize);                                     \
                }                                                                                 \
                nvshmemx_sync_##SC(PE_start, logPE_stride, PE_size, pSync);                       \
            } while (pe_offset < PE_size);                                                        \
        }                                                                                         \
        NVSHMEMI_SYNC_##SC();                                                                     \
    } while (0)

// barrier limited
#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUT_BAR_DIRECT(                                             \
    SC, TYPE, OP, dest, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync)             \
    do {                                                                                           \
        int stride = 1 << logPE_stride;                                                            \
        int next_rank = -1;                                                                        \
        int src_offset = -1;                                                                       \
        int next_offset = -1;                                                                      \
        char *base = NULL;                                                                         \
        char *peer_base = NULL;                                                                    \
        char *peer_source = NULL;                                                                  \
        int counter = 0;                                                                           \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int i;                                                                                     \
                                                                                                   \
        base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +       \
                                      nvshmemi_mype_d));                                            \
        src_offset = ((char *)source - base);                                                      \
                                                                                                   \
        next_rank = (nvshmemi_mype_d + (stride)) % (stride * PE_size);                              \
        next_offset = src_offset;                                                                  \
        peer_base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +  \
                                           next_rank));                                            \
        peer_source = peer_base + next_offset;                                                     \
        GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT_DIRECT(SC, TYPE, OP, source, peer_source, next_rank, \
                                                     dest, counter, nreduce, myIdx, groupSize,     \
                                                     PE_start, logPE_stride, PE_size, pSync);      \
                                                                                                   \
        for (i = 2; i < PE_size; i++) {                                                            \
            next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                      \
            next_offset = src_offset;                                                              \
            peer_base = (char *)((void *)__ldg(                                                    \
                (const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));                \
            peer_source = peer_base + next_offset;                                                 \
            GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT_DIRECT(                                          \
                SC, TYPE, OP, dest, peer_source, next_rank, dest, counter, nreduce, myIdx,         \
                groupSize, PE_start, logPE_stride, PE_size, pSync);                                \
        }                                                                                          \
        NVSHMEMI_SYNC_##SC();                                                                      \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTRING(SC, TYPE, OP, dest, source, nelems, PE_start,       \
                                               logPE_stride, PE_size, pWrk, pSync)                 \
    do {                                                                                           \
        int next_rank = -1;                                                                        \
        int prev_rank = -1;                                                                        \
        int i, j;                                                                                  \
        int stride = 1 << logPE_stride;                                                            \
        int PE_end = PE_start + (stride * PE_size);                                                \
        uint32_t tmp[2];                                                                           \
        long *tmp_rdxn = (long *)gpu_own_intm_rdxn_addr_d;                                         \
        int *tmp_int_rdxn = (int *)((long *)&tmp_rdxn[1]);                                         \
        uint32_t payld;                                                                            \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        volatile uint32_t *my_notif_ptr = NULL;                                                    \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        tmp[1] = 1;                                                                                \
        next_rank = (nvshmemi_mype_d != (PE_end - stride)) ? (nvshmemi_mype_d + stride) : PE_start;  \
        prev_rank = (nvshmemi_mype_d != PE_start) ? (nvshmemi_mype_d - stride) : (PE_end - stride);  \
        my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx); \
                                                                                                   \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                   \
            payld = *((uint32_t *)source + j);                                                     \
            tmp[0] = payld;                                                                        \
            *tmp_rdxn = *((long *)tmp);                                                            \
            nvshmemx_long_signal(((long *)pWrk + j), *tmp_rdxn, next_rank);                      \
        }                                                                                          \
        GPU_HEAD_CHECK_OP_THREADGROUP(TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize);     \
        /* sync needed on volta (intermittent hangs seen otherwise) */                             \
        NVSHMEMI_SYNC_##SC();                                                                      \
                                                                                                   \
        for (i = 1; i < (PE_size - 1); i++) {                                                      \
            __threadfence_system();                                                                \
            /* Don't want notification to overtake data transfer */                                \
            *tmp_int_rdxn = 1;                                                                     \
            nvshmemx_int_signal(((int *)((int *)((long *)pWrk + (nelems * subelems)) + myIdx)),    \
                                *tmp_int_rdxn, prev_rank);                                         \
            while (1 != *my_notif_ptr)                                                             \
                ;                                                                                  \
            *my_notif_ptr = 0;                                                                     \
            for (j = myIdx; j < nelems * subelems; j += groupSize) {                               \
                payld = *((uint32_t *)dest + j);                                                   \
                tmp[0] = payld;                                                                    \
                *tmp_rdxn = *((long *)tmp);                                                        \
                nvshmemx_long_signal(((long *)pWrk + j), *tmp_rdxn, next_rank);                    \
            }                                                                                      \
            GPU_HEAD_CHECK_OP_THREADGROUP(TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize); \
            NVSHMEMI_SYNC_##SC();                                                                  \
        }                                                                                          \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTRING_DIRECT(                                             \
    SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size, pWrk, pSync)              \
    do {                                                                                           \
        int offset;                                                                                \
        char *notif_pwrk_dest;                                                                     \
        char *round_pwrk_dest;                                                                     \
        int next_rank = -1;                                                                        \
        int prev_rank = -1;                                                                        \
        int i, j;                                                                                  \
        int stride = 1 << logPE_stride;                                                            \
        int PE_end = PE_start + (stride * PE_size);                                                \
        uint32_t tmp[2];                                                                           \
        uint32_t payld;                                                                            \
        uint32_t *notif_ptr = NULL;                                                                \
        volatile uint32_t *my_notif_ptr = NULL;                                                    \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        tmp[1] = 1;                                                                                \
        offset =                                                                                   \
            (char *)pWrk - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +   \
                                          nvshmemi_mype_d));                                        \
        my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx); \
        next_rank = (nvshmemi_mype_d != (PE_end - stride)) ? (nvshmemi_mype_d + stride) : PE_start;  \
        round_pwrk_dest =                                                                          \
            (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank)) +    \
            offset;                                                                                \
        prev_rank = (nvshmemi_mype_d != PE_start) ? (nvshmemi_mype_d - stride) : (PE_end - stride);  \
        notif_pwrk_dest =                                                                          \
            (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + prev_rank)) +    \
            offset;                                                                                \
        notif_ptr =                                                                                \
            (uint32_t *)((uint32_t *)((uint64_t *)notif_pwrk_dest + (nelems * subelems)) + myIdx); \
                                                                                                   \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                   \
            payld = *((uint32_t *)source + j);                                                     \
            tmp[0] = payld;                                                                        \
            *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);                               \
        }                                                                                          \
        GPU_HEAD_CHECK_OP_THREADGROUP(TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize);     \
        /* sync needed on volta (intermittent hangs seen otherwise) */                             \
        NVSHMEMI_SYNC_##SC();                                                                      \
                                                                                                   \
        for (i = 1; i < (PE_size - 1); i++) {                                                      \
            __threadfence_system();                                                                \
            /* Don't want notification to overtake data transfer */                                \
            *notif_ptr = 1;                                                                        \
            while (1 != *my_notif_ptr)                                                             \
                ;                                                                                  \
            *my_notif_ptr = 0;                                                                     \
            for (j = myIdx; j < nelems * subelems; j += groupSize) {                               \
                payld = *((uint32_t *)dest + j);                                                   \
                tmp[0] = payld;                                                                    \
                *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);                           \
            }                                                                                      \
            GPU_HEAD_CHECK_OP_THREADGROUP(TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize); \
            NVSHMEMI_SYNC_##SC();                                                                  \
        }                                                                                          \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL(SC, TYPE, OP, dest, source, nelems, PE_start,       \
                                              logPE_stride, PE_size, pWrk, pSync)                 \
    do {                                                                                          \
        int i, j;                                                                                 \
        int stride = 1 << logPE_stride;                                                           \
        int PE_end = PE_start + (stride * PE_size);                                               \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                          \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                    \
        long *tmp_rdxn = (long *)gpu_own_intm_rdxn_addr_d;                                        \
        uint32_t tmp[2];                                                                          \
        uint32_t payld;                                                                           \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                           \
        int my_active_set_pe = ((nvshmemi_mype_d - PE_start) / stride);                            \
        tmp[1] = 1;                                                                               \
                                                                                                  \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                  \
            payld = *((uint32_t *)source + j);                                                    \
            tmp[0] = payld;                                                                       \
            *tmp_rdxn = *((long *)tmp);                                                           \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                                 \
                nvshmemx_long_signal(((long *)pWrk + j + (nelems * subelems * my_active_set_pe)), \
                                     *tmp_rdxn, i);                                               \
            }                                                                                     \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                          \
                nvshmemx_long_signal(((long *)pWrk + j + (nelems * subelems * my_active_set_pe)), \
                                     *tmp_rdxn, i);                                               \
            }                                                                                     \
        }                                                                                         \
        GPU_HEAD_CHECKALL_OP_THREADGROUP(TYPE, OP, dest, pWrk, source, nelems, PE_start, stride,  \
                                         PE_size, myIdx, groupSize);                              \
        __threadfence();                                                                          \
        NVSHMEMI_SYNC_##SC();                                                                     \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL_DIRECT(SC, TYPE, OP, dest, source, nelems, PE_start, \
                                                     logPE_stride, PE_size, pWrk, pSync)           \
    do {                                                                                           \
        int offset;                                                                                \
        char *round_pwrk_dest;                                                                     \
        int i, j;                                                                                  \
        int stride = 1 << logPE_stride;                                                            \
        int PE_end = PE_start + (stride * PE_size);                                                \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        uint32_t tmp[2];                                                                           \
        uint32_t payld;                                                                            \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        int my_active_set_pe = ((nvshmemi_mype_d - PE_start) / stride);                             \
        tmp[1] = 1;                                                                                \
        offset =                                                                                   \
            (char *)pWrk - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +   \
                                          nvshmemi_mype_d));                                        \
                                                                                                   \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                   \
            payld = *((uint32_t *)source + j);                                                     \
            tmp[0] = payld;                                                                        \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                           \
                round_pwrk_dest =                                                                  \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +    \
                    offset;                                                                        \
                *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =      \
                    *((uint64_t *)tmp);                                                            \
            }                                                                                      \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                                  \
                round_pwrk_dest =                                                                  \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +    \
                    offset;                                                                        \
                *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =      \
                    *((uint64_t *)tmp);                                                            \
            }                                                                                      \
        }                                                                                          \
        GPU_HEAD_CHECKALL_OP_THREADGROUP(TYPE, OP, dest, pWrk, source, nelems, PE_start, stride,   \
                                         PE_size, myIdx, groupSize);                               \
        __threadfence();                                                                           \
        NVSHMEMI_SYNC_##SC();                                                                      \
    } while (0)

#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride, \
                                       PE_size, pWrk, pSync)                                       \
    do {                                                                                           \
        NVSHMEMXI_GPU_RDXN_THREADGROUP_ZCOPY_GET_BAR_DIRECT(                                       \
            SC, TYPE, OP, dest, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);    \
    } while (0)
#else
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride, \
                                       PE_size, pWrk, pSync)                                       \
    do {                                                                                           \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        int pwrk_req_sz_allgather = ((subelems * nelems) * sizeof(uint64_t)) * PE_size;            \
        int wrk_size = NVSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);                             \
        if (subelems && (pwrk_req_sz_allgather <= wrk_size)) {                                     \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL_DIRECT(SC, TYPE, OP, dest, source, nreduce,      \
                                                         PE_start, logPE_stride, PE_size, pWrk,    \
                                                         pSync);                                   \
        } else {                                                                                   \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_ZCOPY_GET_BAR_DIRECT(SC, TYPE, OP, dest, source,        \
                                                                nreduce, PE_start, logPE_stride,   \
                                                                PE_size, pWrk, pSync);             \
        }                                                                                          \
    } while (0)
#endif
#else
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride, \
                                       PE_size, pWrk, pSync)                                       \
    do {                                                                                           \
        GPU_RDXN_SEGMENT_THREADGROUP(SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride,   \
                                     PE_size, pWrk, pSync);                                        \
    } while (0)
#else
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPE, OP, dest, source, nelems, PE_start, logPE_stride, \
                                       PE_size, pWrk, pSync)                                       \
    do {                                                                                           \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        int pwrk_req_sz_allgather = ((subelems * nelems) * sizeof(uint64_t)) * PE_size;            \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int pwrk_req_sz_ring =                                                                     \
            ((subelems * nelems) * sizeof(uint64_t)) + (groupSize * sizeof(uint32_t));             \
        int wrk_size = NVSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);                             \
        if (subelems && pwrk_req_sz_allgather <= wrk_size) {                                       \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL(SC, TYPE, OP, dest, source, nreduce, PE_start,   \
                                                  logPE_stride, PE_size, pWrk, pSync);             \
        } else if (subelems && pwrk_req_sz_ring <= wrk_size) {                                     \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTRING(SC, TYPE, OP, dest, source, nreduce, PE_start,  \
                                                   logPE_stride, PE_size, pWrk, pSync);            \
        } else {                                                                                   \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUT_BAR(SC, TYPE, OP, dest, source, nreduce, PE_start,  \
                                                   logPE_stride, PE_size, pWrk, pSync);            \
        }                                                                                          \
    } while (0)
#endif
#endif

#define DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, TYPE, OP)                      \
    __device__ void nvshmemx_##TYPE##_##OP##_to_all_##SC(SRC_DST(TYPE), NR, PS, PL, PZ, \
                                                         PWRK(TYPE), PSYN) {            \
        NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPE, OP, dest, source, nreduce, PE_start,   \
                                       logPE_stride, PE_size, pWrk, pSync);             \
    }

#define DEFN_REDUCE_THREADGROUP(SC)                                 \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, int, and);     \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, long, and);    \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, short, and);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, double, max);  \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, float, max);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, int, max);     \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, long, max);    \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, short, max);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, double, min);  \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, float, min);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, int, min);     \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, long, min);    \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, short, min);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, double, sum);  \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, float, sum);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, int, sum);     \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, long, sum);    \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, short, sum);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, double, prod); \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, float, prod);  \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, int, prod);    \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, long, prod);   \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, short, prod);  \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, int, or);      \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, long, or);     \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, short, or);    \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, int, xor);     \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, long, xor);    \
    DEFN_NVSHMEMX_GPU_TYPE_REDUCE_THREADGROUP_OP(SC, short, xor);

DEFN_REDUCE_THREADGROUP(warp);
DEFN_REDUCE_THREADGROUP(block);

#endif

#define RDXN_ON_STREAM_KERNEL(TYPE, OP)                                                           \
    __global__ void rdxn_##TYPE##_##OP##_on_stream_kernel(                                        \
        TYPE *dest, const TYPE *source, int nreduce, int PE_start, int logPE_stride, int PE_size, \
        TYPE *pWrk, long *pSync) {                                                                \
        if (!blockIdx.x)                                                                          \
            nvshmemx_##TYPE##_##OP##_to_all_block(dest, source, nreduce, PE_start, logPE_stride,  \
                                                  PE_size, pWrk, pSync);                          \
    }

RDXN_ON_STREAM_KERNEL(int, and);
RDXN_ON_STREAM_KERNEL(long, and);
RDXN_ON_STREAM_KERNEL(short, and);
RDXN_ON_STREAM_KERNEL(double, max);
RDXN_ON_STREAM_KERNEL(float, max);
RDXN_ON_STREAM_KERNEL(int, max);
RDXN_ON_STREAM_KERNEL(long, max);
RDXN_ON_STREAM_KERNEL(short, max);
RDXN_ON_STREAM_KERNEL(double, min);
RDXN_ON_STREAM_KERNEL(float, min);
RDXN_ON_STREAM_KERNEL(int, min);
RDXN_ON_STREAM_KERNEL(long, min);
RDXN_ON_STREAM_KERNEL(short, min);
RDXN_ON_STREAM_KERNEL(double, sum);
RDXN_ON_STREAM_KERNEL(float, sum);
RDXN_ON_STREAM_KERNEL(int, sum);
RDXN_ON_STREAM_KERNEL(long, sum);
RDXN_ON_STREAM_KERNEL(short, sum);
RDXN_ON_STREAM_KERNEL(double, prod);
RDXN_ON_STREAM_KERNEL(float, prod);
RDXN_ON_STREAM_KERNEL(int, prod);
RDXN_ON_STREAM_KERNEL(long, prod);
RDXN_ON_STREAM_KERNEL(short, prod);
RDXN_ON_STREAM_KERNEL(int, or);
RDXN_ON_STREAM_KERNEL(long, or);
RDXN_ON_STREAM_KERNEL(short, or);
RDXN_ON_STREAM_KERNEL(int, xor);
RDXN_ON_STREAM_KERNEL(long, xor);
RDXN_ON_STREAM_KERNEL(short, xor);

#define CALL_RDXN_ON_STREAM(TYPE, OP)                                                             \
    extern "C" void call_rdxn_##TYPE##_##OP##_on_stream_kern(                                     \
        TYPE *dest, const TYPE *source, int nreduce, int PE_start, int logPE_stride, int PE_size, \
        TYPE *pWrk, long *pSync, cudaStream_t stream) {                                           \
        int num_threads_per_block =                                                               \
            (MAX_THREADS_PER_CTA > nreduce) ? nreduce : MAX_THREADS_PER_CTA;                      \
        int num_blocks = 1;                                                                       \
        rdxn_##TYPE##_##OP##_on_stream_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(  \
            dest, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, pSync);                 \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                                   \
    }

CALL_RDXN_ON_STREAM(int, and);
CALL_RDXN_ON_STREAM(long, and);
CALL_RDXN_ON_STREAM(short, and);
CALL_RDXN_ON_STREAM(double, max);
CALL_RDXN_ON_STREAM(float, max);
CALL_RDXN_ON_STREAM(int, max);
CALL_RDXN_ON_STREAM(long, max);
CALL_RDXN_ON_STREAM(short, max);
CALL_RDXN_ON_STREAM(double, min);
CALL_RDXN_ON_STREAM(float, min);
CALL_RDXN_ON_STREAM(int, min);
CALL_RDXN_ON_STREAM(long, min);
CALL_RDXN_ON_STREAM(short, min);
CALL_RDXN_ON_STREAM(double, sum);
CALL_RDXN_ON_STREAM(float, sum);
CALL_RDXN_ON_STREAM(int, sum);
CALL_RDXN_ON_STREAM(long, sum);
CALL_RDXN_ON_STREAM(short, sum);
CALL_RDXN_ON_STREAM(double, prod);
CALL_RDXN_ON_STREAM(float, prod);
CALL_RDXN_ON_STREAM(int, prod);
CALL_RDXN_ON_STREAM(long, prod);
CALL_RDXN_ON_STREAM(short, prod);
CALL_RDXN_ON_STREAM(int, or);
CALL_RDXN_ON_STREAM(long, or);
CALL_RDXN_ON_STREAM(short, or);
CALL_RDXN_ON_STREAM(int, xor);
CALL_RDXN_ON_STREAM(long, xor);
CALL_RDXN_ON_STREAM(short, xor);
