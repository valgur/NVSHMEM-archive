/*
 * * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"
#include "cuComplex.h"
#include "nvshmem_internal.h"

#define perform_rd_sum(result, op1, op2) result = op1 + op2
#define perform_rd_prod(result, op1, op2) result = op1 * op2
#define perform_rd_and(result, op1, op2) result = op1 & op2
#define perform_rd_or(result, op1, op2) result = op1 | op2
#define perform_rd_xor(result, op1, op2) result = op1 ^ op2
#define perform_rd_min(result, op1, op2) result = (op1 > op2) ? op2 : op1
#define perform_rd_max(result, op1, op2) result = (op1 > op2) ? op1 : op2

rdxn_fxn_ptr_t rdxn_fptr_arr[rd_op_null][rd_dt_null];
rdxn_comb_fxn_ptr_t rdxn_comb_fptr_arr[rd_op_null][rd_dt_null];

#define DEFN_NVSHMEM_CPU_OP_FXN(TYPE, OP)                                                        \
    __global__ void nvshmemi_rdxn_cpu_##TYPE##_##OP##_to_all(void *x, void *y, void *z, int n) { \
        int i = blockDim.x * blockIdx.x + threadIdx.x;                                           \
        if (i < n) {                                                                             \
            perform_##OP(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));                  \
        }                                                                                        \
    }                                                                                            \
                                                                                                 \
    __global__ void nvshmemi_rdxn_cpu_comb_##TYPE##_##OP##_to_all(int src_off, int dest_off,     \
                                                                  rdxn_opr_t rdx_op) {           \
        int i = blockDim.x * blockIdx.x + threadIdx.x;                                           \
        int j = -1;                                                                              \
        int next_rank = -1;                                                                      \
        int stride = -1;                                                                         \
        stride = 1 << rdx_op.logPE_stride;                                                       \
        next_rank = (nvshmemi_mype_d + stride) % (rdx_op.PE_size * stride);                       \
        TYPE *x = (TYPE *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +          \
                                 nvshmemi_mype_d));                                               \
        x = (TYPE *)((char *)x + src_off);                                                       \
        TYPE *y =                                                                                \
            (TYPE *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));   \
        y = (TYPE *)((char *)y + src_off);                                                       \
        TYPE *z = (TYPE *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +          \
                                 nvshmemi_mype_d));                                               \
        z = (TYPE *)((char *)z + dest_off);                                                      \
        if (i < rdx_op.nreduce) {                                                                \
            perform_##OP(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));                  \
            for (j = 0; j < (rdx_op.PE_size - 2); j++) {                                         \
                next_rank = (next_rank + stride) % (rdx_op.PE_size * stride);                    \
                y = (TYPE *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +        \
                                   next_rank));                                                  \
                y = (TYPE *)((char *)y + src_off);                                               \
                perform_##OP(*((TYPE *)z + i), *((TYPE *)z + i), *((TYPE *)y + i));              \
            }                                                                                    \
        }                                                                                        \
    }

#define DEFN_NVSHMEM_CPU_OP_FXN2(TYPE, TYPE2, OP)                                                 \
    __global__ void nvshmemi_rdxn_cpu_##TYPE##_##TYPE2##_##OP##_to_all(void *x, void *y, void *z, \
                                                                       int n) {                   \
        int i = blockDim.x * blockIdx.x + threadIdx.x;                                            \
        if (i < n) {                                                                              \
            perform_##OP(*((TYPE TYPE2 *)z + i), *((TYPE TYPE2 *)x + i), *((TYPE TYPE2 *)y + i)); \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    __global__ void nvshmemi_rdxn_cpu_comb_##TYPE##_##TYPE2##_##OP##_to_all(                      \
        int src_off, int dest_off, rdxn_opr_t rdx_op) {                                           \
        int i = blockDim.x * blockIdx.x + threadIdx.x;                                            \
        int j = -1;                                                                               \
        int next_rank = -1;                                                                       \
        int stride = -1;                                                                          \
        stride = 1 << rdx_op.logPE_stride;                                                        \
        next_rank = (nvshmemi_mype_d + stride) % (rdx_op.PE_size * stride);                        \
        TYPE TYPE2 *x = (TYPE TYPE2 *)(__ldg(                                                     \
            (const long long unsigned *)nvshmemi_peer_heap_base_d + nvshmemi_mype_d));              \
        x = (TYPE TYPE2 *)((char *)x + src_off);                                                  \
        TYPE TYPE2 *y = (TYPE TYPE2 *)(__ldg(                                                     \
            (const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));                   \
        y = (TYPE TYPE2 *)((char *)y + src_off);                                                  \
        TYPE TYPE2 *z = (TYPE TYPE2 *)(__ldg(                                                     \
            (const long long unsigned *)nvshmemi_peer_heap_base_d + nvshmemi_mype_d));              \
        z = (TYPE TYPE2 *)((char *)z + dest_off);                                                 \
        if (i < rdx_op.nreduce) {                                                                 \
            perform_##OP(*((TYPE TYPE2 *)z + i), *((TYPE TYPE2 *)x + i), *((TYPE TYPE2 *)y + i)); \
            for (j = 0; j < (rdx_op.PE_size - 2); j++) {                                          \
                next_rank = (next_rank + stride) % (rdx_op.PE_size * stride);                     \
                y = (TYPE TYPE2 *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +   \
                                         next_rank));                                             \
                y = (TYPE TYPE2 *)((char *)y + src_off);                                          \
                perform_##OP(*((TYPE TYPE2 *)z + i), *((TYPE TYPE2 *)z + i),                      \
                             *((TYPE TYPE2 *)y + i));                                             \
            }                                                                                     \
        }                                                                                         \
    }

#define DEFN_NVSHMEM_CPU_OP_FXN2_NOT_SUPPORTED(TYPE, TYPE2, OP)                                   \
    __global__ void nvshmemi_rdxn_cpu_##TYPE##_##TYPE2##_##OP##_to_all(void *x, void *y, void *z, \
                                                                       int n) {                   \
        printf("reduction of type " #TYPE "_" #TYPE2 " not supported in CUDA\n");                 \
        assert(0);                                                                                \
    }                                                                                             \
                                                                                                  \
    __global__ void nvshmemi_rdxn_cpu_comb_##TYPE##_##TYPE2##_##OP##_to_all(                      \
        int src_off, int dest_off, rdxn_opr_t rdx_op) {                                           \
        printf("reduction of type " #TYPE "_" #TYPE2 " not supported in CUDA\n");                 \
        assert(0);                                                                                \
    }

#define DEFN_NVSHMEM_CPU_OP_ARTH_FXNS(OP)     \
    DEFN_NVSHMEM_CPU_OP_FXN(double, OP);      \
    DEFN_NVSHMEM_CPU_OP_FXN(float, OP);       \
    DEFN_NVSHMEM_CPU_OP_FXN(int, OP);         \
    DEFN_NVSHMEM_CPU_OP_FXN(long, OP);        \
    DEFN_NVSHMEM_CPU_OP_FXN(short, OP);       \
    DEFN_NVSHMEM_CPU_OP_FXN2(long, long, OP); \
    DEFN_NVSHMEM_CPU_OP_FXN2_NOT_SUPPORTED(long, double, OP);

#define DEFN_NVSHMEM_CPU_OP_EXTRM_FXNS(OP)    \
    DEFN_NVSHMEM_CPU_OP_FXN(double, OP);      \
    DEFN_NVSHMEM_CPU_OP_FXN(float, OP);       \
    DEFN_NVSHMEM_CPU_OP_FXN(int, OP);         \
    DEFN_NVSHMEM_CPU_OP_FXN(long, OP);        \
    DEFN_NVSHMEM_CPU_OP_FXN(short, OP);       \
    DEFN_NVSHMEM_CPU_OP_FXN2(long, long, OP); \
    DEFN_NVSHMEM_CPU_OP_FXN2_NOT_SUPPORTED(long, double, OP);

#define DEFN_NVSHMEM_CPU_OP_LOGICAL_FXNS(OP) \
    DEFN_NVSHMEM_CPU_OP_FXN(short, OP);      \
    DEFN_NVSHMEM_CPU_OP_FXN(int, OP);        \
    DEFN_NVSHMEM_CPU_OP_FXN(long, OP);       \
    DEFN_NVSHMEM_CPU_OP_FXN2(long, long, OP);

DEFN_NVSHMEM_CPU_OP_ARTH_FXNS(rd_sum);
DEFN_NVSHMEM_CPU_OP_ARTH_FXNS(rd_prod);
DEFN_NVSHMEM_CPU_OP_EXTRM_FXNS(rd_min);
DEFN_NVSHMEM_CPU_OP_EXTRM_FXNS(rd_max);
DEFN_NVSHMEM_CPU_OP_LOGICAL_FXNS(rd_and);
DEFN_NVSHMEM_CPU_OP_LOGICAL_FXNS(rd_or);
DEFN_NVSHMEM_CPU_OP_LOGICAL_FXNS(rd_xor);

#define ASSIGN_FPTR(TYPE, OP)                                                                  \
    do {                                                                                       \
        rdxn_fptr_arr[OP][rd_dt_##TYPE] = &nvshmemi_rdxn_cpu_##TYPE##_##OP##_to_all;           \
        rdxn_comb_fptr_arr[OP][rd_dt_##TYPE] = &nvshmemi_rdxn_cpu_comb_##TYPE##_##OP##_to_all; \
    } while (0)

#define ASSIGN_FPTR2(TYPE, TYPE2, OP)                                 \
    do {                                                              \
        rdxn_fptr_arr[OP][rd_dt_##TYPE##_##TYPE2] =                   \
            &nvshmemi_rdxn_cpu_##TYPE##_##TYPE2##_##OP##_to_all;      \
        rdxn_comb_fptr_arr[OP][rd_dt_##TYPE##_##TYPE2] =              \
            &nvshmemi_rdxn_cpu_comb_##TYPE##_##TYPE2##_##OP##_to_all; \
    } while (0)

#define ASSGN_OP_ARTH_FXN_PTRS(OP) \
    ASSIGN_FPTR(double, OP);       \
    ASSIGN_FPTR(float, OP);        \
    ASSIGN_FPTR(int, OP);          \
    ASSIGN_FPTR(long, OP);         \
    ASSIGN_FPTR(short, OP);        \
    ASSIGN_FPTR2(long, long, OP);  \
    ASSIGN_FPTR2(long, double, OP);

#define ASSGN_OP_EXTRM_FXN_PTRS(OP) \
    ASSIGN_FPTR(double, OP);        \
    ASSIGN_FPTR(float, OP);         \
    ASSIGN_FPTR(int, OP);           \
    ASSIGN_FPTR(long, OP);          \
    ASSIGN_FPTR(short, OP);         \
    ASSIGN_FPTR2(long, long, OP);   \
    ASSIGN_FPTR2(long, double, OP);

#define ASSGN_OP_LOGICAL_FXN_PTRS(OP) \
    ASSIGN_FPTR(short, OP);           \
    ASSIGN_FPTR(int, OP);             \
    ASSIGN_FPTR(long, OP);            \
    ASSIGN_FPTR2(long, long, OP);

extern "C" int nvshmemi_rdxn_fxn_ptrs_init() {
    ASSGN_OP_ARTH_FXN_PTRS(rd_sum);
    ASSGN_OP_ARTH_FXN_PTRS(rd_prod);
    ASSGN_OP_EXTRM_FXN_PTRS(rd_min);
    ASSGN_OP_EXTRM_FXN_PTRS(rd_max);
    ASSGN_OP_LOGICAL_FXN_PTRS(rd_and);
    ASSGN_OP_LOGICAL_FXN_PTRS(rd_or);
    ASSGN_OP_LOGICAL_FXN_PTRS(rd_xor);
    return 0;
}

extern "C" int nvshmemi_rdxn_cpu_op_kernel(void *x, void *y, void *z, rdxn_opr_t *rdx_op) {
    int status = 0;
    int num_blocks;

    num_blocks = (rdx_op->nreduce + nvshm_rdx_num_tpb - 1) / nvshm_rdx_num_tpb;

    rdxn_fptr_arr[rdx_op->op_type]
                 [rdx_op
                      ->op_dt_type]<<<num_blocks, nvshm_rdx_num_tpb, 0, nvshmem_state->my_stream>>>(
                     x, y, z, rdx_op->nreduce);
    CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));

    return status;
}

extern "C" int nvshmemxi_rdxn_cpu_op_kernel(void *x, void *y, void *z, rdxn_opr_t *rdx_op) {
    int status = 0;
    int num_blocks;

    num_blocks = (rdx_op->nreduce + nvshm_rdx_num_tpb - 1) / nvshm_rdx_num_tpb;

    rdxn_fptr_arr[rdx_op->op_type]
                 [rdx_op->op_dt_type]<<<num_blocks, nvshm_rdx_num_tpb, 0, rdx_op->stream>>>(
                     x, y, z, rdx_op->nreduce);

    return status;
}

extern "C" int nvshmemi_rdxn_cpu_op_comb_kernel(int src_offset, int dest_offset,
                                                rdxn_opr_t *rdx_op) {
    int status = 0;
    int num_blocks;
    rdxn_opr_t rdx_op_arg;

    num_blocks = (rdx_op->nreduce + nvshm_rdx_num_tpb - 1) / nvshm_rdx_num_tpb;

    memcpy(&rdx_op_arg, rdx_op, sizeof(rdxn_opr_t));

    rdxn_comb_fptr_arr
        [rdx_op->op_type]
        [rdx_op->op_dt_type]<<<num_blocks, nvshm_rdx_num_tpb, 0, nvshmem_state->my_stream>>>(
            src_offset, dest_offset, rdx_op_arg);
    CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));

    return status;
}
