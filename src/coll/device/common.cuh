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

template <typename T, rdxn_ops_t op>
__device__ typename enable_if<is_integral<T>::value, T>::type perform_gpu_rdxn(T op1, T op2) {
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_AND:
            return op1 & op2;
        case RDXN_OPS_OR:
            return op1 | op2;
        case RDXN_OPS_XOR:
            return op1 ^ op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

template <typename T, rdxn_ops_t op>
__device__ typename enable_if<!is_integral<T>::value, T>::type perform_gpu_rdxn(T op1, T op2) {
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

#endif
