/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "cpu_coll.h"

int nvshmemi_reduction_op_cpu_in_kern_ring(rdxn_opr_t *rdx_op) {
    int status = 0;
    int src_offset = -1;
    int dest_offset = -1;
    char *base = NULL;

    base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
    src_offset = (char *)rdx_op->source - base;
    dest_offset = (char *)rdx_op->dest - base;

    nvshmemi_rdxn_cpu_op_comb_kernel(src_offset, dest_offset, rdx_op);

fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_reduction_op_cpu_in_mem_ring(rdxn_opr_t *rdx_op) {
    int status = 0;
    int msg_len = rdx_op->nreduce * rdx_op->op_size;
    int stride = 1 << rdx_op->logPE_stride;
    int next_rank = -1;
    int src_offset = -1;
    int next_offset = -1;
    char *base = NULL;
    char *op1 = NULL, *op2 = NULL;
    int i;
    volatile int *bcast_data_arr = nvshm_cpu_coll_info.cpu_bcast_int_data_arr;

    base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
    src_offset = (char *)rdx_op->source - base;

    if (nvshm_cpu_coll_sync_reqd) {
        status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
        if (status) NVSHMEMI_COLL_CPU_ERR_POP();
    }

    if (nvshm_cpu_coll_offset_reqd) {
        bcast_data_arr[nvshmem_state->mype] = src_offset;
        status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
        if (status) NVSHMEMI_COLL_CPU_ERR_POP();
    }

    CUDA_CHECK(cuMemcpyDtoD((CUdeviceptr)rdx_op->dest, (CUdeviceptr)rdx_op->source, msg_len));
    op1 = (char *)rdx_op->dest;

    for (i = 1; i < rdx_op->PE_size; i++) {
        next_rank = (nvshmem_state->mype + (i * stride)) % (stride * rdx_op->PE_size);
        next_offset = nvshm_cpu_coll_offset_reqd ? bcast_data_arr[next_rank] : src_offset;
        op2 = (char *)nvshmem_state->peer_heap_base[next_rank] + next_offset;
        nvshmemi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
    }

    if (nvshm_cpu_coll_sync_reqd) {
        status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
        if (status) NVSHMEMI_COLL_CPU_ERR_POP();
    }

fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_reduction_op_cpu_p2p_allgather(rdxn_opr_t *rdx_op) {
    int status = 0;
    int msg_len = rdx_op->nreduce * rdx_op->op_size;
    char *op1 = NULL, *op2 = NULL;
    char *tmp_operands;
    int i;

    // allocate memory for operands from all peers

    tmp_operands = (char *)nvshmemi_malloc(msg_len * rdx_op->PE_size);
    if (!tmp_operands) {
        fprintf(stderr, "nvshmemi_malloc failed in p2p_allgather\n");
    }

    // gather operands from all peers

    nvshmemi_collect((void *)tmp_operands, rdx_op->source, rdx_op->op_size, rdx_op->nreduce,
                     rdx_op->PE_start, rdx_op->logPE_stride, rdx_op->PE_size, NULL);

    // perform local reduction

    // alternative 1
    op1 = tmp_operands;
    for (i = 1; i < rdx_op->PE_size; i++) {
        op2 = tmp_operands + (i * msg_len);
        nvshmemi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
    }
    cudaStreamSynchronize(0);

    if (!nvshm_use_p2p_cpu_push) {
        nvshmem_getmem(rdx_op->dest, (void *)tmp_operands, msg_len, nvshmem_state->mype);
    } else {
        nvshmem_putmem(rdx_op->dest, (void *)tmp_operands, msg_len, nvshmem_state->mype);
    }

    // free temporary memory
    nvshmemi_free(tmp_operands);
fn_out:
    return status;
}

int nvshmemi_reduction_op_cpu_p2p_on_demand_gather(rdxn_opr_t *rdx_op) {
    int status = 0;
    int msg_len = rdx_op->nreduce * rdx_op->op_size;
    int stride = 1 << rdx_op->logPE_stride;
    int next_rank = -1;
    char *op1 = NULL, *op2 = NULL;
    char *tmp_operand;
    int i;

    // allocate memory for 1 operand
    tmp_operand = (char *)nvshmemi_malloc(msg_len);

    if (!tmp_operand) {
        fprintf(stderr, "nvshmemi_malloc failed in p2p_allgather\n");
    }

    if (!nvshm_use_p2p_cpu_push) {
        nvshmem_getmem(rdx_op->dest, rdx_op->source, msg_len, nvshmem_state->mype);
    } else {
        nvshmem_putmem(rdx_op->dest, rdx_op->source, msg_len, nvshmem_state->mype);
    }

    // perform local reduction
    for (i = 1; i < rdx_op->PE_size; i++) {
        next_rank = (nvshmem_state->mype + (i * stride)) % (stride * rdx_op->PE_size);
        if (!nvshm_use_p2p_cpu_push) {
            nvshmem_getmem((void *)tmp_operand, rdx_op->source, msg_len, next_rank);
        } else {
            nvshmem_putmem((void *)tmp_operand, rdx_op->source, msg_len, next_rank);
        }
        op1 = (char *)rdx_op->dest;
        op2 = tmp_operand;
        nvshmemi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
        cudaStreamSynchronize(0);
    }

    // free temporary memory
    nvshmemi_free(tmp_operand);

fn_out:
    return status;
}

int nvshmemi_reduction_op_cpu_p2p_segmented_gather(rdxn_opr_t *rdx_op) {
    int status = 0;
    int msg_len = rdx_op->nreduce * rdx_op->op_size;
    int stride = 1 << rdx_op->logPE_stride;
    int next_rank = -1;
    int src_offset = -1;
    int next_offset = -1;
    char *base = NULL;
    char *op1 = NULL, *op2 = NULL;
    char *tmp_operand;
    int i, j;
    int elems_comp = 0;
    int remainder = 0;
    int rnds_floor = 0;
    int offset = 0;
    int exchange_size = 0;

    // allocate memory for a small intermediate buffer space
    tmp_operand = (char *)nvshmemi_malloc(nvshm_cpu_rdxn_seg_size);

    if (!tmp_operand) {
        fprintf(stderr, "nvshmemi_malloc failed in p2p_allgather\n");
    }

    if (!nvshm_use_p2p_cpu_push) {
        nvshmem_getmem(rdx_op->dest, rdx_op->source, msg_len, nvshmem_state->mype);
    } else {
        nvshmem_putmem(rdx_op->dest, rdx_op->source, msg_len, nvshmem_state->mype);
    }

    rnds_floor = (rdx_op->nreduce * rdx_op->op_size) / nvshm_cpu_rdxn_seg_size;
    remainder = (rdx_op->nreduce * rdx_op->op_size) % nvshm_cpu_rdxn_seg_size;

    for (j = 0; j < rnds_floor; j++) {
        exchange_size = nvshm_cpu_rdxn_seg_size;
        // perform local reduction
        for (i = 1; i < rdx_op->PE_size; i++) {
            next_rank = (nvshmem_state->mype + (i * stride)) % (stride * rdx_op->PE_size);
            if (!nvshm_use_p2p_cpu_push) {
                nvshmem_getmem(tmp_operand, (void *)((char *)rdx_op->source + offset),
                               exchange_size, next_rank);
            } else {
                nvshmem_putmem(tmp_operand, (void *)((char *)rdx_op->source + offset),
                               exchange_size, next_rank);
            }
            op1 = (char *)rdx_op->dest + offset;
            op2 = (char *)tmp_operand;
            nvshmemi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
            // TODO: handle this error
            // shmemi_rdxn_cpu_op_kernel(op1, op2, op1,
            //                          (exchange_size / rdx_op->op_size),
            //                          rdx_op->op_size, rdx_op->op_type);
            cudaStreamSynchronize(0);
        }
        offset += exchange_size;
    }

    if (remainder != 0) {
        exchange_size = remainder;
        for (i = 1; i < rdx_op->PE_size; i++) {
            next_rank = (nvshmem_state->mype + (i * stride)) % (stride * rdx_op->PE_size);
            if (!nvshm_use_p2p_cpu_push) {
                nvshmem_getmem(tmp_operand, (void *)((char *)rdx_op->source + offset),
                               exchange_size, next_rank);
            } else {
                nvshmem_putmem(tmp_operand, (void *)((char *)rdx_op->source + offset),
                               exchange_size, next_rank);
            }
            op1 = (char *)rdx_op->dest + offset;
            op2 = (char *)tmp_operand;
            nvshmemi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
            // TODO: handle error
            // shmemi_rdxn_cpu_op_kernel(op1, op2, op1,
            //                          (exchange_size / rdx_op->op_size),
            //                          rdx_op->op_size, rdx_op->op_type);
            cudaStreamSynchronize(0);
        }
    }

    // free temporary memory
    nvshmemi_free(tmp_operand);
fn_out:
    return status;
}

int nvshmemi_rdxn_op_cpu_slxn(rdxn_opr_t *rdx_op) {
    int status = 0;

    if (!nvshm_enable_p2p_cpu_coll) {
        // status = nvshmemi_reduction_op_cpu_in_mem_ring(rdx_op);
        status = nvshmemi_reduction_op_cpu_in_kern_ring(rdx_op);
    } else {
        if (nvshm_use_p2p_cpu_rdxn_allgather) {
            status = nvshmemi_reduction_op_cpu_p2p_allgather(rdx_op);
        } else if (nvshm_use_p2p_cpu_rdxn_od_gather) {
            status = nvshmemi_reduction_op_cpu_p2p_on_demand_gather(rdx_op);
        } else {
            status = nvshmemi_reduction_op_cpu_p2p_segmented_gather(rdx_op);
        }
    }
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_rdxn_r_op_cpu_slxn(rdxn_opr_t *rdx_op) {
    int status = 0;

out:
    return status;
}

int nvshmemi_rdxn_c_op_cpu_slxn(rdxn_opr_t *rdx_op) {
    int status = 0;

out:
    return status;
}

#define DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(TYPE, OP)                                     \
    void nvshmem_##TYPE##_##OP##_to_all(SRC_DST(TYPE), NR, PS, PL, PZ, PWRK(TYPE), PSYN) { \
        if (nvshm_use_tg_for_cpu_coll) {                                                   \
            call_rdxn_##TYPE##_##OP##_on_stream_kern(dest, source, nreduce, PE_start,      \
                                                     logPE_stride, PE_size, pWrk, pSync,   \
                                                     nvshmem_state->my_stream);            \
            CUDA_CHECK(cuStreamSynchronize(nvshmem_state->my_stream));                     \
        } else {                                                                           \
            rdxn_opr_t rd_op;                                                              \
            rd_op.dest = dest;                                                             \
            rd_op.source = source;                                                         \
            rd_op.nreduce = nreduce;                                                       \
            rd_op.PE_start = PE_start;                                                     \
            rd_op.logPE_stride = logPE_stride;                                             \
            rd_op.PE_size = PE_size;                                                       \
            rd_op.pWrk = pWrk;                                                             \
            rd_op.pSync = pSync;                                                           \
            ASSGN_OP_TYPE(TYPE);                                                           \
            rd_op.op_size = sizeof(TYPE);                                                  \
            rd_op.op_type = rd_##OP;                                                       \
            rd_op.stream = 0;                                                              \
            nvshmemi_rdxn_op_cpu_slxn(&rd_op);                                             \
        }                                                                                  \
    }

#define DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(TYPE, TYPE2, OP)                  \
    void nvshmem_##TYPE##TYPE2##_##OP##_to_all(SRC_DST_R(TYPE, TYPE2), NR, PS, PL, PZ, \
                                               PWRK_R(TYPE, TYPE2), PSYN) {            \
        rdxn_opr_t rd_op;                                                              \
        rd_op.dest = dest;                                                             \
        rd_op.source = source;                                                         \
        rd_op.nreduce = nreduce;                                                       \
        rd_op.PE_start = PE_start;                                                     \
        rd_op.logPE_stride = logPE_stride;                                             \
        rd_op.PE_size = PE_size;                                                       \
        rd_op.pWrk = pWrk;                                                             \
        rd_op.pSync = pSync;                                                           \
        rd_op.op_size = sizeof(TYPE TYPE2);                                            \
        ASSGN_OP_TYPE2(TYPE, TYPE2);                                                   \
        rd_op.op_type = rd_##OP;                                                       \
        rd_op.stream = 0;                                                              \
        nvshmemi_rdxn_r_op_cpu_slxn(&rd_op);                                           \
    }

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define DEFN_NVSHMEM_ALL_ARCH_TYPE_C_REDUCE_OP(TYPE, TYPENAME, OP)                                 \
    void nvshmem_##TYPENAME##_##OP##_to_all(SRC_DST_C(TYPE), NR, PS, PL, PZ, PWRK_C(TYPE), PSYN) { \
        rdxn_opr_t rd_op;                                                                          \
        rd_op.dest = dest;                                                                         \
        rd_op.source = source;                                                                     \
        rd_op.nreduce = nreduce;                                                                   \
        rd_op.PE_start = PE_start;                                                                 \
        rd_op.logPE_stride = logPE_stride;                                                         \
        rd_op.PE_size = PE_size;                                                                   \
        rd_op.pWrk = pWrk;                                                                         \
        rd_op.pSync = pSync;                                                                       \
        rd_op.op_size = sizeof(TYPE complex);                                                      \
        ASSGN_OP_TYPE2(TYPE, complex);                                                             \
        rd_op.op_type = rd_##OP;                                                                   \
        rd_op.stream = 0;                                                                          \
        nvshmemi_rdxn_c_op_cpu_slxn(&rd_op);                                                       \
    }
#endif
// and

DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(int, and);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(long, and);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, long, and);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(short, and);

// max

DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(double, max);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(float, max);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(int, max);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(long, max);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, double, max);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, long, max);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(short, max);

// min

DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(double, min);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(float, min);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(int, min);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(long, min);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, double, min);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, long, min);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(short, min);

// sum

#ifdef NVSHMEM_COMPLEX_SUPPORT
DEFN_NVSHMEM_ALL_ARCH_TYPE_C_REDUCE_OP(double, complexd, sum);
DEFN_NVSHMEM_ALL_ARCH_TYPE_C_REDUCE_OP(float, complexf, sum);
#endif
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(double, sum);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(float, sum);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(int, sum);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(long, sum);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, double, sum);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, long, sum);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(short, sum);

// prod

#ifdef NVSHMEM_COMPLEX_SUPPORT
DEFN_NVSHMEM_ALL_ARCH_TYPE_C_REDUCE_OP(double, complexd, prod);
DEFN_NVSHMEM_ALL_ARCH_TYPE_C_REDUCE_OP(float, complexf, prod);
#endif
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(double, prod);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(float, prod);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(int, prod);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(long, prod);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, double, prod);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, long, prod);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(short, prod);

// or

DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(int, or);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(long, or);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, long, or);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(short, or);

// xor

DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(int, xor);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(long, xor);
DEFN_NVSHMEM_ALL_ARCH_TYPE_R_REDUCE_OP_INNER(long, long, xor);
DEFN_NVSHMEM_ALL_ARCH_TYPE_REDUCE_OP(short, xor);
