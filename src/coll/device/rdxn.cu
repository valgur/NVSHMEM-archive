/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"

#ifdef __CUDA_ARCH__

#define GPU_HEAD_CHECK_OP(TYPE, OP, dest, src, actual_src, nelems)                    \
    do {                                                                              \
        int i, k;                                                                     \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                               \
        volatile uint32_t *header = NULL;                                             \
        TYPE tmp;                                                                     \
        uint32_t *tmp_ptr = (uint32_t *)&tmp;                                         \
        uint32_t *payload = NULL;                                                     \
        for (i = 0; i < nelems; i++) {                                                \
            for (k = 0; k < subelems; k++) {                                          \
                payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k);         \
                header = (uint32_t *)payload + 1;                                     \
                while (1 != *header)                                                  \
                    ;                                                                 \
                *header = 0;                                                          \
                *(tmp_ptr + k) = *payload;                                            \
            }                                                                         \
            perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)actual_src + i), tmp); \
        }                                                                             \
    } while (0)

#define GPU_HEAD_CHECKALL_OP(TYPE, OP, dest, src, actual_src, nelems, PE_start, stride, PE_size) \
    do {                                                                                         \
        int i, j, k;                                                                             \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                          \
        volatile uint32_t *header = NULL;                                                        \
        TYPE tmp;                                                                                \
        uint32_t *tmp_ptr = (uint32_t *)&tmp;                                                    \
        uint32_t *payload = NULL;                                                                \
        int my_active_set_pe = ((nvshmemi_mype_d - PE_start) / stride);                           \
        TYPE *src_ptr = (TYPE *)actual_src;                                                      \
                                                                                                 \
        for (j = (my_active_set_pe - 1); j >= 0; j--) {                                          \
            for (i = 0; i < nelems; i++) {                                                       \
                for (k = 0; k < subelems; k++) {                                                 \
                    payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k +                \
                                           (nelems * subelems * j));                             \
                    header = (uint32_t *)payload + 1;                                            \
                    while (1 != *header)                                                         \
                        ;                                                                        \
                    *header = 0;                                                                 \
                    *(tmp_ptr + k) = *payload;                                                   \
                }                                                                                \
                perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);           \
            }                                                                                    \
            src_ptr = dest;                                                                      \
        }                                                                                        \
        for (j = PE_size - 1; j > my_active_set_pe; j--) {                                       \
            for (i = 0; i < nelems; i++) {                                                       \
                for (k = 0; k < subelems; k++) {                                                 \
                    payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k +                \
                                           (nelems * subelems * j));                             \
                    header = (uint32_t *)payload + 1;                                            \
                    while (1 != *header)                                                         \
                        ;                                                                        \
                    *header = 0;                                                                 \
                    *(tmp_ptr + k) = *payload;                                                   \
                }                                                                                \
                perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);           \
            }                                                                                    \
            src_ptr = dest;                                                                      \
        }                                                                                        \
    } while (0)

#define GPU_LINEAR_REDUCE(TYPE, OP, x, y, z, nelems)                                   \
    do {                                                                               \
        int i;                                                                         \
        for (i = 0; i < nelems; i++) {                                                 \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i)); \
        }                                                                              \
    } while (0)

#define GPU_RDXN_ON_DEMAND(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size, pWrk, \
                           pSync)                                                                 \
    do {                                                                                          \
        int stride = 1 << logPE_stride;                                                           \
        int next_rank = -1;                                                                       \
        TYPE *op1 = NULL, *op2 = NULL;                                                            \
        int i;                                                                                    \
        volatile TYPE *tmp_operand;                                                               \
                                                                                                  \
        tmp_operand = (TYPE *)gpu_own_intm_rdxn_addr_d;                                           \
                                                                                                  \
        nvshmem_##TYPE##_put((TYPE *)dest, (TYPE *)source, nelems, nvshmemi_mype_d);               \
        for (i = 1; i < PE_size; i++) {                                                           \
            next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                     \
            nvshmem_##TYPE##_put_nbi((TYPE *)tmp_operand, (TYPE *)source, nelems, next_rank);     \
            nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                              \
            op1 = (TYPE *)dest;                                                                   \
            op2 = (TYPE *)tmp_operand;                                                            \
            GPU_LINEAR_REDUCE(TYPE, OP, op1, op2, op1, nelems);                                   \
            nvshmem_sync(PE_start, logPE_stride, PE_size, pSync);                                 \
        }                                                                                         \
    } while (0)

#define GPU_RDXN_SEGMENT(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size, pWrk, \
                         pSync)                                                                 \
    do {                                                                                        \
        int type_size = sizeof(TYPE);                                                           \
        int msg_len = nelems * type_size;                                                       \
        int stride = 1 << logPE_stride;                                                         \
        int next_rank = -1;                                                                     \
        TYPE *op1 = NULL, *op2 = NULL;                                                          \
        int i, j;                                                                               \
        volatile TYPE *tmp_operand;                                                             \
        int remainder = 0;                                                                      \
        int rnds_floor = 0;                                                                     \
        int offset = 0;                                                                         \
        int exchange_size = 0;                                                                  \
        int nvshm_gpu_rdxn_seg_size = gpu_coll_env_params_var_d.gpu_intm_rdxn_size;             \
                                                                                                \
        tmp_operand = (TYPE *)gpu_own_intm_rdxn_addr_d;                                         \
        nvshmem_##TYPE##_put((TYPE *)dest, (TYPE *)source, nelems, nvshmemi_mype_d);             \
                                                                                                \
        rnds_floor = msg_len / nvshm_gpu_rdxn_seg_size;                                         \
        remainder = msg_len % nvshm_gpu_rdxn_seg_size;                                          \
                                                                                                \
        for (j = 0; j < rnds_floor; j++) {                                                      \
            exchange_size = nvshm_gpu_rdxn_seg_size;                                            \
            for (i = 1; i < PE_size; i++) {                                                     \
                next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);               \
                nvshmem_##TYPE##_put_nbi((TYPE *)tmp_operand, (TYPE *)source + offset,          \
                                         (exchange_size / sizeof(TYPE)), next_rank);            \
                nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                        \
                op1 = (TYPE *)dest + offset;                                                    \
                op2 = (TYPE *)tmp_operand;                                                      \
                GPU_LINEAR_REDUCE(TYPE, OP, op1, op2, op1, (exchange_size / sizeof(TYPE)));     \
                nvshmem_sync(PE_start, logPE_stride, PE_size, pSync);                           \
            }                                                                                   \
            offset += (exchange_size / sizeof(TYPE));                                           \
        }                                                                                       \
                                                                                                \
        if (remainder != 0) {                                                                   \
            exchange_size = remainder;                                                          \
            for (i = 1; i < PE_size; i++) {                                                     \
                next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);               \
                nvshmem_##TYPE##_put_nbi((TYPE *)tmp_operand, (TYPE *)source + offset,          \
                                         (exchange_size / sizeof(TYPE)), next_rank);            \
                nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                        \
                op1 = (TYPE *)dest + offset;                                                    \
                op2 = (TYPE *)tmp_operand;                                                      \
                GPU_LINEAR_REDUCE(TYPE, OP, op1, op2, op1, (exchange_size / sizeof(TYPE)));     \
                nvshmem_sync(PE_start, logPE_stride, PE_size, pSync);                           \
            }                                                                                   \
        }                                                                                       \
                                                                                                \
    } while (0)

#define GPU_RDXN_ZCOPY_GET_BAR(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size,   \
                               pWrk, pSync)                                                       \
    do {                                                                                          \
        int stride = 1 << logPE_stride;                                                           \
        int next_rank = -1;                                                                       \
        int src_offset = -1;                                                                      \
        int next_offset = -1;                                                                     \
        char *base = NULL;                                                                        \
        char *peer_base = NULL;                                                                   \
        char *peer_source = NULL;                                                                 \
        int i;                                                                                    \
                                                                                                  \
        base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +      \
                                      nvshmemi_mype_d));                                           \
        src_offset = ((char *)source - base);                                                     \
                                                                                                  \
        next_rank = (nvshmemi_mype_d + (stride)) % (stride * PE_size);                             \
        next_offset = src_offset;                                                                 \
        peer_base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                           next_rank));                                           \
        peer_source = peer_base + next_offset;                                                    \
        GPU_LINEAR_REDUCE(TYPE, OP, (void *)source, peer_source, dest, nreduce);                  \
        nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                                  \
                                                                                                  \
        for (i = 2; i < PE_size; i++) {                                                           \
            next_rank = (nvshmemi_mype_d + (i * stride)) % (stride * PE_size);                     \
            next_offset = src_offset;                                                             \
            peer_base = (char *)((void *)__ldg(                                                   \
                (const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));               \
            peer_source = peer_base + next_offset;                                                \
            GPU_LINEAR_REDUCE(TYPE, OP, dest, peer_source, dest, nreduce);                        \
            nvshmem_barrier(PE_start, logPE_stride, PE_size, pSync);                              \
        }                                                                                         \
    } while (0)

#define NVSHMEMI_GPU_RDXN_PUTALL(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size, \
                                 pWrk, pSync)                                                     \
    do {                                                                                          \
        int offset;                                                                               \
        int i, j;                                                                                 \
        int stride = 1 << logPE_stride;                                                           \
        int PE_end = PE_start + (stride * PE_size);                                               \
        uint32_t tmp[2];                                                                          \
        uint32_t payld;                                                                           \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                           \
        int my_active_set_pe = ((nvshmemi_mype_d - PE_start) / stride);                            \
        tmp[1] = 1;                                                                               \
                                                                                                  \
        for (j = 0; j < nelems * subelems; j++) {                                                 \
            payld = *((uint32_t *)source + j);                                                    \
            tmp[0] = payld;                                                                       \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                                 \
                nvshmemx_long_signal((long *)pWrk + j + (nelems * subelems * my_active_set_pe),   \
                                     *((long *)tmp), i);                                          \
            }                                                                                     \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                          \
                nvshmemx_long_signal((long *)pWrk + j + (nelems * subelems * my_active_set_pe),   \
                                     *((long *)tmp), i);                                          \
            }                                                                                     \
        }                                                                                         \
        GPU_HEAD_CHECKALL_OP(TYPE, OP, dest, pWrk, source, nelems, PE_start, stride, PE_size);    \
        __threadfence();                                                                          \
    } while (0)

#define NVSHMEMI_GPU_RDXN_PUTALL_DIRECT(TYPE, OP, dest, source, nelems, PE_start, logPE_stride,  \
                                        PE_size, pWrk, pSync)                                    \
    do {                                                                                         \
        int offset;                                                                              \
        char *round_pwrk_dest;                                                                   \
        int i, j;                                                                                \
        int stride = 1 << logPE_stride;                                                          \
        int PE_end = PE_start + (stride * PE_size);                                              \
        uint32_t tmp[2];                                                                         \
        uint32_t payld;                                                                          \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                          \
        int my_active_set_pe = ((nvshmemi_mype_d - PE_start) / stride);                           \
        tmp[1] = 1;                                                                              \
        offset =                                                                                 \
            (char *)pWrk - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + \
                                          nvshmemi_mype_d));                                      \
                                                                                                 \
        for (j = 0; j < nelems * subelems; j++) {                                                \
            payld = *((uint32_t *)source + j);                                                   \
            tmp[0] = payld;                                                                      \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                         \
                round_pwrk_dest =                                                                \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +  \
                    offset;                                                                      \
                *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =    \
                    *((uint64_t *)tmp);                                                          \
            }                                                                                    \
            for (i = PE_start; i < nvshmemi_mype_d; i += stride) {                                \
                round_pwrk_dest =                                                                \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +  \
                    offset;                                                                      \
                *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =    \
                    *((uint64_t *)tmp);                                                          \
            }                                                                                    \
        }                                                                                        \
        GPU_HEAD_CHECKALL_OP(TYPE, OP, dest, pWrk, source, nelems, PE_start, stride, PE_size);   \
        __threadfence();                                                                         \
    } while (0)

#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMI_GPU_RDXN(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size, pWrk, \
                          pSync)                                                                 \
    do {                                                                                         \
        GPU_RDXN_ZCOPY_GET_BAR(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size,  \
                               pWrk, pSync);                                                     \
    } while (0)
#else
#define NVSHMEMI_GPU_RDXN(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size, pWrk, \
                          pSync)                                                                 \
    do {                                                                                         \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                          \
        int pwrk_req_sz_allgather = ((subelems * nelems) * sizeof(uint64_t)) * PE_size;          \
        /*int pwrk_req_sz_ring = ((subelems * nelems) * sizeof(uint64_t));*/                     \
        int wrk_size = NVSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);                           \
        if (subelems && pwrk_req_sz_allgather <= wrk_size) {                                     \
            NVSHMEMI_GPU_RDXN_PUTALL_DIRECT(TYPE, OP, dest, source, nelems, PE_start,            \
                                            logPE_stride, PE_size, pWrk, pSync);                 \
        } else {                                                                                 \
            GPU_RDXN_ZCOPY_GET_BAR(TYPE, OP, dest, source, nelems, PE_start, logPE_stride,       \
                                   PE_size, pWrk, pSync);                                        \
        }                                                                                        \
    } while (0)
#endif
#else
#define NVSHMEMI_GPU_RDXN(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size, pWrk,   \
                          pSync)                                                                   \
    do {                                                                                           \
        int k = gpu_coll_env_params_var_d.reduce_recexch_kval;                                     \
        if (PE_start == 0 && logPE_stride == 0 && PE_size == nvshmemi_npes_d &&                     \
            NVSHMEM_REDUCE_MIN_WRKDATA_SIZE >=                                                     \
                ((k - 1) * nelems + k * reduce_recexch_step2_nphases_d * nelems + reduce_recexch_step2_nphases_d * nelems)) { \
            GPU_RDXN_RECEXCH(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size,      \
                             pWrk, pSync);                                                         \
        } else {                                                                                   \
            if (gpu_coll_env_params_var_d.gpu_intm_rdxn_size >= (nelems * sizeof(TYPE))) {         \
                GPU_RDXN_ON_DEMAND(TYPE, OP, dest, source, nelems, PE_start, logPE_stride,         \
                                   PE_size, pWrk, pSync);                                          \
            } else {                                                                               \
                GPU_RDXN_SEGMENT(TYPE, OP, dest, source, nelems, PE_start, logPE_stride, PE_size,  \
                                 pWrk, pSync);                                                     \
            }                                                                                      \
        }                                                                                          \
    } while (0)
#endif

#define DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(TYPE, OP)                                                 \
    __device__ void nvshmem_##TYPE##_##OP##_to_all(SRC_DST(TYPE), NR, PS, PL, PZ, PWRK(TYPE),     \
                                                   PSYN) {                                        \
        NVSHMEMI_GPU_RDXN(TYPE, OP, dest, source, nreduce, PE_start, logPE_stride, PE_size, pWrk, \
                          pSync);                                                                 \
    }

/* pWrk usage - (k - 1) * nreduce for step 1
              - k * step2_nphases * nreduce for receiving step 2 data
              - step2_nphases * nreduce for sending data of each phase */
#define GPU_RDXN_RECEXCH(TYPE, OP, dst, source, nreduce, PE_start, logPE_stride, PE_size, pWrk,    \
                         pSync)                                                                    \
    do {                                                                                           \
        int step1_sendto = reduce_recexch_step1_sendto_d;                                                         \
        int step1_nrecvs = reduce_recexch_step1_nrecvs_d;                                                         \
        int *step1_recvfrom = reduce_recexch_step1_recvfrom_d;                                                    \
        int step2_nphases = reduce_recexch_step2_nphases_d;                                                       \
        int **step2_nbrs = reduce_recexch_step2_nbrs_d;                                                           \
        int rank = nvshmemi_mype_d;                                                                 \
        int k = gpu_coll_env_params_var_d.reduce_recexch_kval;                                     \
                                                                                                   \
        int in_step2 = (step1_sendto == -1); /* whether this rank participates in Step 2 */        \
                                                                                                   \
        if (in_step2 == 1) {                                                                       \
            for (int i = 0; i < nreduce; i++) {                                                    \
                dst[i] = source[i];                                                                \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (in_step2 == 0) {                                                                       \
            int offset = (step1_sendto - rank - 1) * nreduce;                                      \
            nvshmem_##TYPE##_put_nbi(pWrk + offset, source, nreduce, step1_sendto);                \
            nvshmem_fence();                                                                       \
            nvshmemx_long_signal(pSync + rank, !NVSHMEM_SYNC_VALUE, step1_sendto);                 \
        } else if (step1_nrecvs != 0) {                                                            \
            nvshmem_long_wait_until_all(pSync + step1_recvfrom[step1_nrecvs - 1], step1_nrecvs,    \
                                        NULL, NVSHMEM_CMP_EQ, !NVSHMEM_SYNC_VALUE);                \
            for (int i = 0; i < step1_nrecvs; i++) {                                               \
                int offset = (rank - step1_recvfrom[i] - 1) * nreduce;                             \
                GPU_LINEAR_REDUCE(TYPE, OP, dst, (pWrk + offset), dst, nreduce);                   \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        /* Step 2 */                                                                               \
        if (in_step2) {                                                                            \
            int send_offset = (k - 1) * nreduce + k * step2_nphases * nreduce;                     \
            int recv_offset = (k - 1) * nreduce;                                                   \
            for (int phase = 0; phase < step2_nphases; phase++) {                                  \
                int num_small = k - 1;                                                             \
                for (int i = 0; i < k - 1; i++) {                                                  \
                    if (step2_nbrs[phase][i] > rank) {                                             \
                        num_small = i;                                                             \
                        break;                                                                     \
                    }                                                                              \
                }                                                                                  \
                /* copy the data to end of pWrk that can be used as source for puts                \
                   while we use dst for reduction */                                               \
                for (int i = 0; i < nreduce; i++) {                                                \
                    pWrk[send_offset + phase * nreduce + i] = dst[i];                              \
                }                                                                                  \
                                                                                                   \
                for (int i = 0; i < k - 1; i++) {                                                  \
                    int offset = recv_offset + k * phase * nreduce + num_small * nreduce;          \
                    nvshmem_##TYPE##_put_nbi(pWrk + offset, pWrk + send_offset + phase * nreduce,  \
                                             nreduce, step2_nbrs[phase][i]);                       \
                }                                                                                  \
                nvshmem_fence();                                                                   \
                for (int i = 0; i < k - 1; i++) {                                                  \
                    nvshmemx_long_signal(pSync + rank, NVSHMEM_SYNC_VALUE + 1,                     \
                                         step2_nbrs[phase][i]);                                    \
                }                                                                                  \
                                                                                                   \
                for (int i = 0; i < k - 1; i++) {                                                  \
                    nvshmem_long_wait_until(pSync + step2_nbrs[phase][i], NVSHMEM_CMP_EQ,          \
                                            NVSHMEM_SYNC_VALUE + 1);                               \
                    int offset = recv_offset + k * phase * nreduce;                                \
                    if (step2_nbrs[phase][i] < rank)                                               \
                        offset += i * nreduce;                                                     \
                    else                                                                           \
                        offset += (i + 1) * nreduce;                                               \
                    GPU_LINEAR_REDUCE(TYPE, OP, dst, (pWrk + offset), dst, nreduce);               \
                }                                                                                  \
                /*nvshmem_quiet(); */ /*wait for my puts to complete */                            \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        /* Step 3 */                                                                               \
        if (step1_nrecvs > 0) {                                                                    \
            for (int i = 0; i < step1_nrecvs; i++) {                                               \
                nvshmem_##TYPE##_put_nbi(dst, dst, nreduce, step1_recvfrom[i]);                    \
                nvshmem_fence();                                                                   \
                nvshmemx_long_signal(pSync + rank, NVSHMEM_SYNC_VALUE + 1, step1_recvfrom[i]);     \
            }                                                                                      \
        } else if (step1_sendto != -1) {                                                           \
            nvshmem_long_wait_until(pSync + step1_sendto, NVSHMEM_CMP_EQ, NVSHMEM_SYNC_VALUE + 1); \
        }                                                                                          \
                                                                                                   \
        for (int i = 0; i < nvshmemi_npes_d; i++)                                                   \
            pSync[i] = NVSHMEM_SYNC_VALUE; /* should this be a volatile write? */                  \
                                                                                                   \
    } while (0);
// and

DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(int, and);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(long, and);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(short, and);

// max

DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(double, max);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(float, max);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(int, max);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(long, max);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(short, max);

// min

DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(double, min);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(float, min);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(int, min);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(long, min);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(short, min);

// sum

DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(double, sum);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(float, sum);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(int, sum);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(long, sum);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(short, sum);

// prod

DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(double, prod);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(float, prod);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(int, prod);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(long, prod);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(short, prod);

// or

DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(int, or);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(long, or);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(short, or);

// xor

DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(int, xor);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(long, xor);
DEFN_NVSHMEM_GPU_TYPE_REDUCE_OP(short, xor);

#endif
