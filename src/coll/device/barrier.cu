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

extern __constant__ int barrier_dissem_kval_d;
__device__ void nvshmem_sync_algo(int, int, int, volatile long *, volatile long *, bool);

#ifdef __CUDA_ARCH__

#define NVSHMEMI_GET_ACTUAL_RANK(rank, effective_root, PE_start, PE_size, stride)       \
    do {                                                                                \
        rank = ((rank + effective_root) < PE_size) ? (rank + effective_root)            \
                                                   : (rank + effective_root - PE_size); \
        rank *= stride;                                                                 \
        rank += PE_start;                                                               \
    } while (0)


template<int k, int logk>
__device__ __inline__ void sync_dissem_pow2(int PE_start, int logPE_stride, int PE_size,
                                               volatile long *pSync, volatile long *counter,
                                               bool is_barrier_all) {
    int my_idx_in_active_set = (nvshmemi_mype_d - PE_start) >> (logPE_stride);
    volatile long *sync_arr;
    if (!is_barrier_all)
        sync_arr = (volatile long *)pSync + (counter[0] & 1) * nvshmemi_npes_d;
    else
        sync_arr = (volatile long *)pSync;
    int shift;
    int to_nbr_idx, to_nbr;
    int from_nbr_idx, from_nbr;

    int temp = PE_size - 1; /* used to calculate number of phases */
    int pow_k = 1;
    int phase_num = 0;
    while(temp) {
        /* notify neighbors */
        for (int j = 1; j <= k - 1; j++) {
            shift = j << phase_num;
            if (shift >= PE_size) break;
            to_nbr_idx = my_idx_in_active_set + shift;
            if (to_nbr_idx >= PE_size) to_nbr_idx = to_nbr_idx - PE_size;
            to_nbr = PE_start + (to_nbr_idx << logPE_stride);

            nvshmemx_long_signal(((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr);
        }

        /* wait for neighbors notification */
        for (int j = 1; j <= k - 1; j++) {
            shift = j << phase_num;
            if (shift >= PE_size) break;
            from_nbr_idx = my_idx_in_active_set - shift;
            if (from_nbr_idx < 0) from_nbr_idx = PE_size + from_nbr_idx;
            from_nbr = PE_start + (from_nbr_idx << logPE_stride);

            nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER);
            if (!is_barrier_all)
                *(sync_arr + from_nbr) = NVSHMEM_SYNC_VALUE;
        }

        pow_k <<= logk;
        temp >>= logk;
        phase_num += 1;
    }

    counter[0] += 1;
}


template __device__ void sync_dissem_pow2<2, 1>(int, int, int, volatile long *, volatile long *, bool);
template __device__ void sync_dissem_pow2<4, 2>(int, int, int, volatile long *, volatile long *, bool);
template __device__ void sync_dissem_pow2<8, 3>(int, int, int, volatile long *, volatile long *, bool);
template __device__ void sync_dissem_pow2<16, 4>(int, int, int, volatile long *, volatile long *, bool);
template __device__ void sync_dissem_pow2<32, 5>(int, int, int, volatile long *, volatile long *, bool);

__device__ __inline__ void sync_dissem(int PE_start, int logPE_stride, int PE_size,
                                          volatile long *pSync, volatile long *counter,
                                          bool is_barrier_all) {
    int stride = 1 << logPE_stride;
    int num_phases = 0;
    int k = min(barrier_dissem_kval_d, PE_size); /* radix for the dissemination algorithm */
    int my_idx_in_active_set = (nvshmemi_mype_d - PE_start) / stride;
    volatile long *sync_arr;
    if (!is_barrier_all)
        sync_arr = (volatile long *)pSync + (counter[0] & 1) * nvshmemi_npes_d;
    else
        sync_arr = (volatile long *)pSync;
    int shift;
    int to_nbr_idx, to_nbr;
    int from_nbr_idx, from_nbr;

    int temp = PE_size - 1; /* used to calculate number of phases */
    while (temp) {
        num_phases++;
        temp /= k;
    }

    int pow_k = 1;
    for (int i = 0; i < num_phases; i++) {
        /* notify neighbors */
        for (int j = 1; j <= k - 1; j++) {
            shift = j * pow_k;
            if (shift >= PE_size) break;
            to_nbr_idx = (my_idx_in_active_set + shift) % PE_size;
            to_nbr = PE_start + to_nbr_idx * stride;

            nvshmemx_long_signal(((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr);
        }

        /* wait for neighbors notification */
        for (int j = 1; j <= k - 1; j++) {
            shift = j * pow_k;
            if (shift >= PE_size) break;

            from_nbr_idx = my_idx_in_active_set - shift;
            if (from_nbr_idx < 0) from_nbr_idx = PE_size + from_nbr_idx;
            from_nbr = PE_start + from_nbr_idx * stride;

            nvshmemi_wait_until_greater_than_equals<volatile long>(sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER);
            if (!is_barrier_all)
                *(sync_arr + from_nbr) = NVSHMEM_SYNC_VALUE;
        }
        pow_k *= k;
    }

    counter[0] += 1;
}

__device__ void nvshmem_sync_algo(int PE_start, int logPE_stride, int PE_size, volatile long *pSync, volatile long *counter, bool is_barrier_all) {
    int k = min(barrier_dissem_kval_d, PE_size); /* radix for the dissemination algorithm */
    k = max(k, 2);
    switch (k) {
        case 2:
            sync_dissem_pow2<2, 1>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);
            break;
        case 4:
            sync_dissem_pow2<4, 2>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);
            break;
        case 8:
            sync_dissem_pow2<8, 3>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);
            break;
        case 16:
            sync_dissem_pow2<16, 4>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);
            break;
        case 32:
            sync_dissem_pow2<32, 5>(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);
            break;
        default:
            sync_dissem(PE_start, logPE_stride, PE_size, pSync, counter, is_barrier_all);
            break;
    }
}

__device__ void nvshmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    nvshmem_quiet();

    nvshmem_sync_algo(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);

    if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_LDST) {
        __threadfence(); /* To ensure barrier algo is complete before issuing consistency op */
        nvshmemi_proxy_enforce_consistency_at_target_no_membar();
    }
    __threadfence(); /* To prevent reordering of instructions after barrier */
}

__device__ void nvshmem_barrier_all() {
    nvshmem_quiet();

    nvshmem_sync_algo(0, 0, nvshmemi_npes_d, gpu_ipsync_d, gpu_icounter_d, 1);

    if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_LDST) {
        __threadfence(); /* To ensure barrier algo is complete before issuing consistency op */
        nvshmemi_proxy_enforce_consistency_at_target_no_membar();
    }
    __threadfence(); /* To prevent reordering of instructions after barrier */
}

__device__ void nvshmem_sync(int PE_start, int logPE_stride, int PE_size, long *pSync) {
    nvshmem_sync_algo(PE_start, logPE_stride, PE_size, pSync, gpu_icounter_barrier_d, 0);
}

__device__ void nvshmem_sync_all() {
    nvshmem_sync_algo(0, 0, nvshmemi_npes_d, gpu_ipsync_d, gpu_icounter_d, 1);
}
#endif
