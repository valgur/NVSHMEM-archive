/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_team.h"
#include "nvshmemi_coll.h"

#ifdef __CUDA_ARCH__

typedef enum rdxn_ops {
    RDXN_OPS_AND = 0,
    RDXN_OPS_OR,
    RDXN_OPS_XOR,
    RDXN_OPS_MIN,
    RDXN_OPS_MAX,
    RDXN_OPS_SUM,
    RDXN_OPS_PROD
} rdxn_ops_t;

template <typename TYPE>
__device__ static inline void
gpu_payload_wait_for_ready(TYPE *src, int pe_idx, int , int nelems)
{
    TYPE tmp;
    volatile uint32_t *header = NULL;
    uint32_t *tmp_ptr = (uint32_t *)&tmp;
    uint32_t *payload = NULL;
    int subelems = sizeof(TYPE) / sizeof(uint32_t);
    int subelem_idx;

    for (subelem_idx = 0; i < subelems; subelem_idx++) {
        payload = (uint32_t *)((uint64_t *)src + (elem_idx * subelems) + subelem_idx + (nelems * subelems * pe_idx));
        header = (uint32_t *)payload + 1;
        while (1 != *header)
            ;
        *header = 0;
        *(tmp_ptr + subelem_idx) = *payload;
    }
}

template <typename TYPE>
__device__ static inline void
gpu_head_checkall_op(rdxn_ops_t op, TYPE *dest, const TYPE *src, TYPE *actual_src,
                     int nelems, int start, int stride, int size)
{
    int i, j;
    int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);
    TYPE *src_ptr = (TYPE *)actual_src;

    switch (op) {
        case RDXN_OPS_AND:
            for (j = (my_active_set_pe - 1); j >= 0; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i, nelems);
                    perform_gpu_rd_and(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            for (j = size - 1; j > my_active_set_pe; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i, nelems);
                    perform_gpu_rd_and(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            break;
        case RDXN_OPS_OR:
            for (j = (my_active_set_pe - 1); j >= 0; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_or(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            for (j = size - 1; j > my_active_set_pe; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_or(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            break;
        case RDXN_OPS_XOR:
            for (j = (my_active_set_pe - 1); j >= 0; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_xor(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            for (j = size - 1; j > my_active_set_pe; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_xor(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            break;
        case RDXN_OPS_MIN:
            for (j = (my_active_set_pe - 1); j >= 0; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_min(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            for (j = size - 1; j > my_active_set_pe; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_min(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            break;
        case RDXN_OPS_MAX:
            for (j = (my_active_set_pe - 1); j >= 0; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_max(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            for (j = size - 1; j > my_active_set_pe; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_max(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            break;
        case RDXN_OPS_SUM:
            for (j = (my_active_set_pe - 1); j >= 0; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_sum(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            for (j = size - 1; j > my_active_set_pe; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_sum(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            break;
        case RDXN_OPS_PROD:
            for (j = (my_active_set_pe - 1); j >= 0; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_prod(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            for (j = size - 1; j > my_active_set_pe; j--) {
                for (i = 0; i < nelems; i++) {
                    gpu_payload_wait_for_ready(src, j, i);
                    perform_gpu_rd_prod(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);
                }
                src_ptr = dest;
            }
            break;
    }
}

template <typename TYPE>
__device__ static inline void
gpu_linear_reduce(rdxn_ops_t op, TYPE *x, TYPE *y, TYPE *z, int nelems)
{
    int i;

    switch (op) {
        case RDXN_OPS_AND:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_and(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));
            }
            break;
        case RDXN_OPS_OR:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_or(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));
            }
            break;
        case RDXN_OPS_XOR:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_xor(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));
            }
            break;
        case RDXN_OPS_MIN:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_min(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));
            }
            break;
        case RDXN_OPS_MAX:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_max(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));
            }
            break;
        case RDXN_OPS_SUM:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_sum(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));
            }
            break;
        case RDXN_OPS_PROD:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_prod(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i));
            }
            break;
    }
}

template <>
__device__ inline void
gpu_linear_reduce<double>(rdxn_ops_t op, double *x, double *y, double *z, int nelems)
{
    int i;

    switch (op) {
        case RDXN_OPS_MIN:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_min(*((double *)z + i), *((double *)x + i), *((double *)y + i));
            }
            break;
        case RDXN_OPS_MAX:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_max(*((double *)z + i), *((double *)x + i), *((double *)y + i));
            }
            break;
        case RDXN_OPS_SUM:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_sum(*((double *)z + i), *((double *)x + i), *((double *)y + i));
            }
            break;
        case RDXN_OPS_PROD:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_prod(*((double *)z + i), *((double *)x + i), *((double *)y + i));
            }
            break;
    }
}

template <>
__device__ inline void
gpu_linear_reduce<float>(rdxn_ops_t op, float *x, float *y, float *z, int nelems)
{
    int i;

    switch (op) {
        case RDXN_OPS_MIN:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_min(*((float *)z + i), *((float *)x + i), *((float *)y + i));
            }
            break;
        case RDXN_OPS_MAX:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_max(*((float *)z + i), *((float *)x + i), *((float *)y + i));
            }
            break;
        case RDXN_OPS_SUM:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_sum(*((float *)z + i), *((float *)x + i), *((float *)y + i));
            }
            break;
        case RDXN_OPS_PROD:
            for (i = 0; i < nelems; i++) {
                perform_gpu_rd_prod(*((float *)z + i), *((float *)x + i), *((float *)y + i));
            }
            break;
    }
}

template <typename TYPE>
__device__ static inline void
gpu_rdxn_on_demand(rdxn_ops_t op, TYPE *dest, const TYPE *source, int nelems,
                   int start, int stride, int size, TYPE * pWrk, long *pSync)
{
    int next_rank = -1;
    TYPE *op1 = NULL, *op2 = NULL;
    int i;
    volatile TYPE *tmp_operand;
    int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);

    tmp_operand = (TYPE *) pWrk;

    put <TYPE> (dest, source, nelems, nvshmemi_mype_d);
    long counter = NVSHMEMI_SYNC_VALUE + 1;
    for (i = 1; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        put_nbi <TYPE> ((TYPE *)tmp_operand, source, nelems, next_rank);
        nvshmemi_barrier(start, stride, size, pSync, &counter);
        op1 = (TYPE *)dest;
        op2 = (TYPE *)tmp_operand;
        gpu_linear_reduce(op, op1, op2, op1, nelems);
        nvshmemi_sync(start, stride, size, pSync, &counter);
    }
    int end = start + size * stride;
    for(i = 0; i < end; i++)
        pSync[i] = NVSHMEMI_SYNC_VALUE;
}

template <typename TYPE>
__device__ static inline void
gpu_rdxn_segment(rdxn_ops_t op, TYPE *dest, const TYPE *source, int nelems,
                   int start, int stride, int size, TYPE * pWrk, long *pSync)
{
    int type_size = sizeof(TYPE);
    int msg_len = nelems * type_size;
    int next_rank = -1;
    TYPE *op1 = NULL, *op2 = NULL;
    int i, j;
    volatile TYPE *tmp_operand;
    int remainder = 0;
    int rnds_floor = 0;
    int offset = 0;
    int exchange_size = 0;
    int nvshm_gpu_rdxn_seg_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE;
    int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);

    tmp_operand = pWrk;
    put <TYPE> (dest, source, nelems, nvshmemi_mype_d);

    rnds_floor = msg_len / nvshm_gpu_rdxn_seg_size;
    remainder = msg_len % nvshm_gpu_rdxn_seg_size;
    long counter = 1;
    for (j = 0; j < rnds_floor; j++) {
        exchange_size = nvshm_gpu_rdxn_seg_size;
        for (i = 1; i < size; i++) {
            next_rank = start + ((my_active_set_pe + i) % size) * stride;
            put_nbi <TYPE> ((TYPE *)tmp_operand, source + offset,
                                        (exchange_size / sizeof(TYPE)), next_rank);
            nvshmemi_barrier(start, stride, size, pSync, &counter);
            op1 = dest + offset;
            op2 = (TYPE *)tmp_operand;
            gpu_linear_reduce(op, op1, op2, op1, (exchange_size / sizeof(TYPE)));
            nvshmemi_sync(start, stride, size, pSync, &counter);
        }
        offset += (exchange_size / sizeof(TYPE));
    }

    if (remainder != 0) {
        exchange_size = remainder;
        for (i = 1; i < size; i++) {
            next_rank = start + ((my_active_set_pe + i) % size) * stride;
            put_nbi <TYPE> ((TYPE *)tmp_operand, source + offset,
                                        (exchange_size / sizeof(TYPE)), next_rank);
            nvshmemi_barrier(start, stride, size, pSync, &counter);
            op1 = dest + offset;
            op2 = (TYPE *)tmp_operand;
            gpu_linear_reduce(op, op1, op2, op1, (exchange_size / sizeof(TYPE)));
            nvshmemi_sync(start, stride, size, pSync, &counter);
        }
    }
    for(i = 0; i < NVSHMEMI_REDUCE_SYNC_SIZE; i++)
        pSync[i] = NVSHMEMI_SYNC_VALUE;
}

template <typename TYPE>
__device__ static inline void
gpu_rdxn_zcopy_get_bar(rdxn_ops_t op, TYPE *dest, const TYPE *source, int nreduce,
                       int start, int stride, int size, TYPE * pWrk, long *pSync)
{
    int next_rank = -1;
    int src_offset = -1;
    int next_offset = -1;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    int i;

    base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + nvshmemi_mype_d));
    src_offset = ((char *)source - base);

    next_rank = (nvshmemi_mype_d + (stride)) % (stride * size);
    next_offset = src_offset;
    peer_base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));
    peer_source = peer_base + next_offset;
    gpu_linear_reduce(op, (void *)source, peer_source, dest, nreduce);
    long counter  = 1;
    nvshmemi_barrier(start, stride, size, pSync, &counter);

    for (i = 2; i < size; i++) {
        next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * size);
        next_offset = src_offset;
        peer_base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));
        peer_source = peer_base + next_offset;
        gpu_linear_reduce(op, dest, peer_source, dest, nreduce);
        nvshmemi_barrier(start, stride, size, pSync, &counter);
    }
    for(i = 0; i < NVSHMEMI_REDUCE_SYNC_SIZE; i++)
        pSync[i] = NVSHMEMI_SYNC_VALUE;
}

template <typename TYPE>
__device__ static inline void
nvshmem_gpu_rdxn_putall_direct(rdxn_ops_t op, TYPE *dest, const TYPE *source, int nelems,
                                int start, int stride, int size, TYPE * pWrk, long *pSync)
{
    int offset;
    char *round_pwrk_dest;
    int i, j;
    int end = start + (stride * size);
    uint32_t tmp[2];
    uint32_t payld;
    int subelems = sizeof(TYPE) / sizeof(uint32_t);
    int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);
    tmp[1] = 1;
    offset = (char *)pWrk - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + nvshmemi_mype_d));

    for (j = 0; j < nelems * subelems; j++) {
        payld = *((uint32_t *)source + j);
        tmp[0] = payld;
        for (i = nvshmemi_mype_d + stride; i < end; i += stride) {
            round_pwrk_dest = (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) + offset;
            *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) = *((uint64_t *)tmp);
        }
        for (i = start; i < nvshmemi_mype_d; i += stride) {
            round_pwrk_dest = (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) + offset;
            *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) = *((uint64_t *)tmp);
        }
    }
    gpu_head_checkall_op(op, dest, pWrk, source, nelems, start, stride, size);
    __threadfence();
}

/* pWrk usage - (k - 1) * nreduce for step 1
              - k * step2_nphases * nreduce for receiving step 2 data
              - step2_nphases * nreduce for sending data of each phase */

template <typename TYPE>
__device__ static inline void
gpu_rdxn_recexch(rdxn_ops_t op, TYPE *dst, const TYPE *source, int nreduce,
                  int start, int stride, int size, TYPE * pWrk, long *pSync)
{
    int step1_sendto = reduce_recexch_step1_sendto_d;
    int step1_nrecvs = reduce_recexch_step1_nrecvs_d;
    int *step1_recvfrom = reduce_recexch_step1_recvfrom_d;
    int step2_nphases = reduce_recexch_step2_nphases_d;
    int **step2_nbrs = reduce_recexch_step2_nbrs_d;
    int rank = nvshmemi_mype_d;
    int k = gpu_coll_env_params_var_d.reduce_recexch_kval;

    int in_step2 = (step1_sendto == -1); /* whether this rank participates in Step 2 */

    if (in_step2 == 1) {
        for (int i = 0; i < nreduce; i++) {
            dst[i] = source[i];
        }
    }

    if (in_step2 == 0) {
        int offset = (step1_sendto - rank - 1) * nreduce;
        put_nbi <TYPE> (pWrk + offset, source, nreduce, step1_sendto);
        nvshmem_fence();
        nvshmemx_long_signal(pSync + rank, !NVSHMEMI_SYNC_VALUE, step1_sendto);
    } else if (step1_nrecvs != 0) {
        nvshmem_long_wait_until_all(pSync + step1_recvfrom[step1_nrecvs - 1], step1_nrecvs,
                                    NULL, NVSHMEM_CMP_EQ, !NVSHMEMI_SYNC_VALUE);
        for (int i = 0; i < step1_nrecvs; i++) {
            int offset = (rank - step1_recvfrom[i] - 1) * nreduce;
            gpu_linear_reduce(op, dst, (pWrk + offset), dst, nreduce);
        }
    }

    /* Step 2 */
    if (in_step2) {
        int send_offset = (k - 1) * nreduce + k * step2_nphases * nreduce;
        int recv_offset = (k - 1) * nreduce;
        for (int phase = 0; phase < step2_nphases; phase++) {
            int num_small = k - 1;
            for (int i = 0; i < k - 1; i++) {
                if (step2_nbrs[phase][i] > rank) {
                    num_small = i;
                    break;
                }
            }
            /* copy the data to end of pWrk that can be used as source for puts
                while we use dst for reduction */
            for (int i = 0; i < nreduce; i++) {
                pWrk[send_offset + phase * nreduce + i] = dst[i];
            }
            for (int i = 0; i < k - 1; i++) {
                int offset = recv_offset + k * phase * nreduce + num_small * nreduce;
                put_nbi <TYPE> (pWrk + offset, pWrk + send_offset + phase * nreduce,
                                nreduce, step2_nbrs[phase][i]);
            }
            nvshmem_fence();
            for (int i = 0; i < k - 1; i++) {
                nvshmemx_long_signal(pSync + rank, NVSHMEMI_SYNC_VALUE + 1, step2_nbrs[phase][i]);
            }

            for (int i = 0; i < k - 1; i++) {
                nvshmem_long_wait_until(pSync + step2_nbrs[phase][i], NVSHMEM_CMP_EQ,
                                        NVSHMEMI_SYNC_VALUE + 1);
                int offset = recv_offset + k * phase * nreduce;
                if (step2_nbrs[phase][i] < rank)
                    offset += i * nreduce;
                else
                    offset += (i + 1) * nreduce;
                gpu_linear_reduce(op, dst, (pWrk + offset), dst, nreduce);
            }
            /*nvshmem_quiet(); */ /*wait for my puts to complete */
        }
    }

    /* Step 3 */
    if (step1_nrecvs > 0) {
        for (int i = 0; i < step1_nrecvs; i++) {
            put_nbi <TYPE> (dst, dst, nreduce, step1_recvfrom[i]);
            nvshmem_fence();
            nvshmemx_long_signal(pSync + rank, NVSHMEMI_SYNC_VALUE + 1, step1_recvfrom[i]);
        }
    } else if (step1_sendto != -1) {
        nvshmem_long_wait_until(pSync + step1_sendto, NVSHMEM_CMP_EQ, NVSHMEMI_SYNC_VALUE + 1);
    }

    for (int i = 0; i < nvshmemi_npes_d; i++)
        pSync[i] = NVSHMEMI_SYNC_VALUE; /* should this be a volatile write? */
}

template <typename TYPE>
__device__ static inline void
nvshmemi_gpu_rdxn(rdxn_ops_t op, TYPE *dest, const TYPE *source, int nelems,
                  int start, int stride, int size, TYPE * pWrk, long *pSync)
{
#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
    gpu_rdxn_zcopy_get_bar(op, dest, source, nelems, start, stride, size, pWrk, pSync);
#else
    int subelems = sizeof(TYPE) / sizeof(uint32_t);
    int pwrk_req_sz_allgather = ((subelems * nelems) * sizeof(uint64_t)) * size;
    /*int pwrk_req_sz_ring = ((subelems * nelems) * sizeof(uint64_t));*/
    int wrk_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);
    if (subelems && pwrk_req_sz_allgather <= wrk_size) {
        nvshmem_gpu_rdxn_putall_direct(op, dest, source, nelems, start,
                                        stride, size, pWrk, pSync);
    } else {
        gpu_rdxn_zcopy_get_bar(op, dest, source, nelems, start, stride, size, pWrk, pSync);
    }
#endif
#else
    int k = gpu_coll_env_params_var_d.reduce_recexch_kval;
    if (start == 0 && stride == 1 && size == nvshmemi_npes_d &&
        NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE >=
        ((k - 1) * nelems + k * reduce_recexch_step2_nphases_d * nelems + reduce_recexch_step2_nphases_d * nelems)) {
        gpu_rdxn_recexch(op, dest, source, nelems, start, stride, size, pWrk, pSync);
    } else {
        if (NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE >= (nelems * sizeof(TYPE))) {
            gpu_rdxn_on_demand(op, dest, source, nelems, start, stride, size, pWrk, pSync);
        } else {
            gpu_rdxn_segment(op, dest, source, nelems, start, stride, size, pWrk, pSync);
        }
    }
#endif
}

#define DEFINE_NVSHMEMI_OP_REDUCE(OP, ENUM)                                                      \
template <typename TYPE>                                                                         \
__device__ inline void nvshmemi_##OP##_reduce(TYPE *dest, const TYPE *source, int nreduce,       \
                                                int start, int stride, int size, TYPE * pWrk,    \
                                                long *pSync) {                                   \
    nvshmemi_gpu_rdxn(RDXN_OPS_##ENUM, dest, source, nreduce, start, stride, size, pWrk, pSync); \
}

DEFINE_NVSHMEMI_OP_REDUCE(and, AND);
DEFINE_NVSHMEMI_OP_REDUCE(or, OR);
DEFINE_NVSHMEMI_OP_REDUCE(xor, XOR);
DEFINE_NVSHMEMI_OP_REDUCE(min, MIN);
DEFINE_NVSHMEMI_OP_REDUCE(max, MAX);
DEFINE_NVSHMEMI_OP_REDUCE(sum, SUM);
DEFINE_NVSHMEMI_OP_REDUCE(prod, PROD);

#define DEFN_NVSHMEM_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                 \
    __device__ int nvshmem_##TYPENAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest,      \
                                                      const TYPE *source, size_t nreduce) { \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[team];                                \
        TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);                        \
        long *pSync = (long *)((long *)pWrk + NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE);             \
        nvshmem_barrier(team);                                                              \
        nvshmemi_##OP##_reduce <TYPE> (dest, source, nreduce, teami->start, teami->stride,  \
                                       teami->size, pWrk, pSync);                           \
        return 0;                                                                           \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, prod)

#undef DEFN_NVSHMEM_TYPENAME_OP_REDUCE
#endif
