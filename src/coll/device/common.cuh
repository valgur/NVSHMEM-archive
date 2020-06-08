/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#ifndef _GPU_COMMON_H_
#define _GPU_COMMON_H_

#define GPU_DT_COPY(TYPE, dest, src, nelems)          \
    do {                                              \
        int i;                                        \
        for (i = 0; i < nelems; i++) {                \
            *((TYPE *)dest + i) = *((TYPE *)src + i); \
        }                                             \
    } while (0)

#define GPU_BITS_COPY_DIRECT(SUFFIX, dest, src, nelems)                       \
    do {                                                                      \
        int i;                                                                \
        for (i = 0; i < nelems; i++) {                                        \
            *((uint##SUFFIX##_t *)dest + i) = *((uint##SUFFIX##_t *)src + i); \
        }                                                                     \
    } while (0)

#define GPU_BITS_COPY_THREADGROUP_DIRECT(SUFFIX, dest, src, nelems, myIdx, groupSize) \
    do {                                                                              \
        int i;                                                                        \
        for (i = myIdx; i < nelems; i += groupSize) {                                 \
            *((uint##SUFFIX##_t *)dest + i) = *((uint##SUFFIX##_t *)src + i);         \
        }                                                                             \
    } while (0)

#define GPU_BITS_REG_COPY(SUFFIX, dest, src, nelems)            \
    do {                                                        \
        int i;                                                  \
        uint64_t tmp;                                           \
        uint32_t *header = NULL;                                \
        uint##SUFFIX##_t *payload = (uint##SUFFIX##_t *)(&tmp); \
        header = ((uint32_t *)payload + 1);                     \
        *header = 1;                                            \
        for (i = 0; i < nelems; i++) {                          \
            *payload = *((uint##SUFFIX##_t *)src + i);          \
            *((uint64_t *)dest + i) = tmp;                      \
        }                                                       \
    } while (0)

#define GPU_BITS_REG_CHECK(SUFFIX, dest, src, nelems)    \
    do {                                                 \
        int i;                                           \
        int subelems = (SUFFIX / 32);                    \
        volatile uint32_t *header = NULL;                \
        uint32_t *payload = NULL;                        \
        for (i = 0; i < nelems * subelems; i++) {        \
            payload = (uint32_t *)((uint64_t *)src + i); \
            header = (uint32_t *)payload + 1;            \
            while (1 != *header)                         \
                ;                                        \
            *((uint32_t *)dest + i) = *payload;          \
            *((uint64_t *)src + i) = NVSHMEM_SYNC_VALUE; \
        }                                                \
    } while (0)

#define GPU_BITS_REG_CHECK_THREADGROUP(SUFFIX, dest, src, nelems, myIdx, groupSize) \
    do {                                                                            \
        int i;                                                                      \
        int subelems = (SUFFIX / 32);                                               \
        volatile uint32_t *header = NULL;                                           \
        uint32_t *payload = NULL;                                                   \
        for (i = myIdx; i < nelems * subelems; i += groupSize) {                    \
            payload = (uint32_t *)((uint64_t *)src + i);                            \
            header = (uint32_t *)payload + 1;                                       \
            while (1 != *header)                                                    \
                ;                                                                   \
            *((uint32_t *)dest + i) = *payload;                                     \
            *((uint64_t *)src + i) = NVSHMEM_SYNC_VALUE;                            \
        }                                                                           \
    } while (0)

#define perform_gpu_rd_sum(result, op1, op2) result = op1 + op2
#define perform_gpu_rd_prod(result, op1, op2) result = op1 * op2
#define perform_gpu_rd_and(result, op1, op2) result = op1 & op2
#define perform_gpu_rd_or(result, op1, op2) result = op1 | op2
#define perform_gpu_rd_xor(result, op1, op2) result = op1 ^ op2
#define perform_gpu_rd_min(result, op1, op2) result = (op1 > op2) ? op2 : op1
#define perform_gpu_rd_max(result, op1, op2) result = (op1 > op2) ? op1 : op2

#endif
