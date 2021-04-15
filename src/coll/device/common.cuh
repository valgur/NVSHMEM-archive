/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _GPU_COMMON_H_
#define _GPU_COMMON_H_

#define GPU_BITS_COPY_THREADGROUP_DIRECT(TYPENAME, TYPE, dest, src, nelems, myIdx, groupSize) \
    do {                                                                              \
        int i;                                                                        \
        for (i = myIdx; i < nelems; i += groupSize) {                                 \
            *((TYPE *)dest + i) = *((TYPE *)src + i);                                 \
        }                                                                             \
    } while (0)

#define perform_gpu_rd_sum(result, op1, op2) result = op1 + op2
#define perform_gpu_rd_prod(result, op1, op2) result = op1 * op2
#define perform_gpu_rd_and(result, op1, op2) result = op1 & op2
#define perform_gpu_rd_or(result, op1, op2) result = op1 | op2
#define perform_gpu_rd_xor(result, op1, op2) result = op1 ^ op2
#define perform_gpu_rd_min(result, op1, op2) result = (op1 > op2) ? op2 : op1
#define perform_gpu_rd_max(result, op1, op2) result = (op1 > op2) ? op1 : op2

#endif
