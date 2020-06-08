/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "cpu_coll.h"

int nvshmemxi_reduction_op_cpu_in_mem_ring_on_stream(rdxn_opr_t *rdx_op) {
    int status = 0;
    int msg_len = rdx_op->nreduce * rdx_op->op_size;
    int stride = 1 << rdx_op->logPE_stride;
    int next_rank = -1;
    int src_offset = -1;
    int next_offset = -1;
    char *base = NULL;
    char *op1 = NULL, *op2 = NULL;
    int i;

    base = (char *)nvshmem_state->peer_heap_base[nvshmem_state->mype];
    src_offset = (char *)rdx_op->source - base;

    status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);

    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

    nvshm_cpu_coll_info.cpu_bcast_int_data_arr[nvshmem_state->mype] = src_offset;
    status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

    CUDA_CHECK(cuMemcpyDtoDAsync((CUdeviceptr)rdx_op->dest, (CUdeviceptr)rdx_op->source, msg_len,
                                 rdx_op->stream));

    op1 = (char *)rdx_op->dest;

    for (i = 1; i < rdx_op->PE_size; i++) {
        next_rank = (nvshmem_state->mype + (i * stride)) % (stride * rdx_op->PE_size);
        next_offset = nvshm_cpu_coll_info.cpu_bcast_int_data_arr[next_rank];
        op2 = (char *)nvshmem_state->peer_heap_base[next_rank] + next_offset;
        nvshmemxi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
    }

    nvshmemx_barrier_on_stream(rdx_op->PE_start, rdx_op->logPE_stride, rdx_op->PE_size,
                               rdx_op->pSync, rdx_op->stream);

    // status = nvshmem_state->boot_handle.barrier(&nvshmem_state->boot_handle);
    // if (status) NVSHMEMI_COLL_CPU_ERR_POP();

fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemxi_reduction_op_cpu_p2p_segmented_gather_on_stream(rdxn_opr_t *rdx_op) {
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
        nvshmemx_getmem_on_stream(rdx_op->dest, rdx_op->source, msg_len, nvshmem_state->mype,
                                  rdx_op->stream);
    } else {
        nvshmemx_putmem_on_stream(rdx_op->dest, rdx_op->source, msg_len, nvshmem_state->mype,
                                  rdx_op->stream);
    }

    rnds_floor = (rdx_op->nreduce * rdx_op->op_size) / nvshm_cpu_rdxn_seg_size;
    remainder = (rdx_op->nreduce * rdx_op->op_size) % nvshm_cpu_rdxn_seg_size;

    for (j = 0; j < rnds_floor; j++) {
        exchange_size = nvshm_cpu_rdxn_seg_size;
        // perform local reduction
        for (i = 1; i < rdx_op->PE_size; i++) {
            next_rank = (nvshmem_state->mype + (i * stride)) % (stride * rdx_op->PE_size);
            if (!nvshm_use_p2p_cpu_push) {
                nvshmemx_getmem_on_stream(tmp_operand, (void *)((char *)rdx_op->source + offset),
                                          exchange_size, next_rank, rdx_op->stream);
            } else {
                nvshmemx_putmem_on_stream(tmp_operand, (void *)((char *)rdx_op->source + offset),
                                          exchange_size, next_rank, rdx_op->stream);
            }
            op1 = (char *)rdx_op->dest + offset;
            op2 = (char *)tmp_operand;
            nvshmemxi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
        }
        offset += exchange_size;
    }

    if (remainder != 0) {
        exchange_size = remainder;
        for (i = 1; i < rdx_op->PE_size; i++) {
            next_rank = (nvshmem_state->mype + (i * stride)) % (stride * rdx_op->PE_size);
            if (!nvshm_use_p2p_cpu_push) {
                nvshmemx_getmem_on_stream(tmp_operand, (void *)((char *)rdx_op->source + offset),
                                          exchange_size, next_rank, rdx_op->stream);
            } else {
                nvshmemx_putmem_on_stream(tmp_operand, (void *)((char *)rdx_op->source + offset),
                                          exchange_size, next_rank, rdx_op->stream);
            }
            op1 = (char *)rdx_op->dest + offset;
            op2 = (char *)tmp_operand;
            nvshmemxi_rdxn_cpu_op_kernel(op1, op2, op1, rdx_op);
        }
    }

    nvshmemx_barrier_on_stream(rdx_op->PE_start, rdx_op->logPE_stride, rdx_op->PE_size,
                               rdx_op->pSync, rdx_op->stream);

    // free temporary memory
    nvshmemi_free(tmp_operand);
fn_out:
    return status;
}

int nvshmemxi_rdxn_op_cpu_slxn_on_stream(rdxn_opr_t *rdx_op) {
    int status = 0;

    if (!nvshm_enable_p2p_cpu_coll) {
        status = nvshmemxi_reduction_op_cpu_in_mem_ring_on_stream(rdx_op);
    } else {
        status = nvshmemxi_reduction_op_cpu_p2p_segmented_gather_on_stream(rdx_op);
    }
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

fn_out:
    return status;
fn_fail:
    return status;
}

#define DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(TYPE, OP)                                 \
    void nvshmemx_##TYPE##_##OP##_to_all_on_stream(SRC_DST(TYPE), NR, PS, PL, PZ, PWRK(TYPE),     \
                                                   PSYN, CS) {                                    \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                           \
        if (nvshm_use_tg_for_stream_coll) {                                                       \
            call_rdxn_##TYPE##_##OP##_on_stream_kern(dest, source, nreduce, PE_start,             \
                                                     logPE_stride, PE_size, pWrk, pSync, stream); \
        } else {                                                                                  \
            rdxn_opr_t rd_op;                                                                     \
            rd_op.dest = dest;                                                                    \
            rd_op.source = source;                                                                \
            rd_op.nreduce = nreduce;                                                              \
            rd_op.PE_start = PE_start;                                                            \
            rd_op.logPE_stride = logPE_stride;                                                    \
            rd_op.PE_size = PE_size;                                                              \
            rd_op.pWrk = pWrk;                                                                    \
            rd_op.pSync = pSync;                                                                  \
            rd_op.op_size = sizeof(TYPE);                                                         \
            ASSGN_OP_TYPE(TYPE);                                                                  \
            rd_op.op_type = rd_##OP;                                                              \
            rd_op.stream = stream;                                                                \
            nvshmemxi_rdxn_op_cpu_slxn_on_stream(&rd_op);                                         \
        }                                                                                         \
    }

#define DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(TYPE, TYPE2, OP)                  \
    void nvshmemx_##TYPE##TYPE2##_##OP##_to_all_on_stream(SRC_DST_R(TYPE, TYPE2), NR, PS, PL, PZ, \
                                                          PWRK_R(TYPE, TYPE2), PSYN, CS) {        \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                           \
        rdxn_opr_t rd_op;                                                                         \
        rd_op.dest = dest;                                                                        \
        rd_op.source = source;                                                                    \
        rd_op.nreduce = nreduce;                                                                  \
        rd_op.PE_start = PE_start;                                                                \
        rd_op.logPE_stride = logPE_stride;                                                        \
        rd_op.PE_size = PE_size;                                                                  \
        rd_op.pWrk = pWrk;                                                                        \
        rd_op.pSync = pSync;                                                                      \
        rd_op.op_size = sizeof(TYPE TYPE2);                                                       \
        ASSGN_OP_TYPE2(TYPE, TYPE2);                                                              \
        rd_op.op_type = rd_##OP;                                                                  \
        rd_op.stream = stream;                                                                    \
        nvshmemxi_rdxn_op_cpu_slxn_on_stream(&rd_op);                                             \
    }

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define DEFN_NVSHMEMX_ALL_ARCH_TYPE_C_REDUCE_OP_ON_STREAM(TYPE, TYPENAME, OP)           \
    void nvshmemx_##TYPENAME##_##OP##_to_all_on_stream(SRC_DST_C(TYPE), NR, PS, PL, PZ, \
                                                       PWRK_C(TYPE), PSYN, CS) {        \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                 \
        rdxn_opr_t rd_op;                                                               \
        rd_op.dest = dest;                                                              \
        rd_op.source = source;                                                          \
        rd_op.nreduce = nreduce;                                                        \
        rd_op.PE_start = PE_start;                                                      \
        rd_op.logPE_stride = logPE_stride;                                              \
        rd_op.PE_size = PE_size;                                                        \
        rd_op.pWrk = pWrk;                                                              \
        rd_op.pSync = pSync;                                                            \
        rd_op.op_size = sizeof(TYPE complex);                                           \
        ASSGN_OP_TYPE2(TYPE, complex);                                                  \
        rd_op.op_type = rd_##OP;                                                        \
        rd_op.stream = stream;                                                          \
        nvshmemxi_rdxn_op_cpu_slxn_on_stream(&rd_op);                                   \
    }
#endif
// and

DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(int, and);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(long, and);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(short, and);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, long, and);

// max

DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(double, max);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(float, max);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(int, max);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(long, max);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(short, max);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, double, max);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, long, max);

// min

DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(double, min);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(float, min);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(int, min);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(long, min);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(short, min);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, double, min);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, long, min);

// sum

#ifdef NVSHMEM_COMPLEX_SUPPORT
DEFN_NVSHMEMX_ALL_ARCH_TYPE_C_REDUCE_OP_ON_STREAM(double, complexd, sum);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_C_REDUCE_OP_ON_STREAM(float, complexf, sum);
#endif
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(double, sum);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(float, sum);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(int, sum);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(long, sum);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(short, sum);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, double, sum);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, long, sum);

// prod

#ifdef NVSHMEM_COMPLEX_SUPPORT
DEFN_NVSHMEMX_ALL_ARCH_TYPE_C_REDUCE_OP_ON_STREAM(double, complexd, prod);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_C_REDUCE_OP_ON_STREAM(float, complexf, prod);
#endif
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(double, prod);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(float, prod);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(int, prod);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(long, prod);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(short, prod);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, double, prod);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, long, prod);

// or

DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(int, or);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(long, or);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(short, or);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, long, or);

// xor

DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(int, xor);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(long, xor);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_REDUCE_OP_ON_STREAM(short, xor);
DEFN_NVSHMEMX_ALL_ARCH_TYPE_R_REDUCE_OP_ON_STREAM_INNER(long, long, xor);
