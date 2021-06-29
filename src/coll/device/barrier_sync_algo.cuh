/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef _BARRIER_SYNC_ALGO_CUH_
#define _BARRIER_SYNC_ALGO_CUH_
#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_util.h"
#include "nvshmemi_coll.h"
#include "nvshmem_common.cuh"
#include "gpu_coll.h"

#ifdef __CUDA_ARCH__

#define NVSHMEMI_SYNC_DISSEM_POW2_ALGO(SC, SC_SUFFIX, SC_PREFIX)                              \
    template <int k, int logk>                                                                \
    __device__ static inline void sync_dissem_pow2##SC_SUFFIX(                                \
        int start, int stride, int size, volatile long *pSync, volatile long *sync_counter) { \
        int my_idx_in_active_set = (nvshmemi_device_state_d.mype - start) / stride;           \
        volatile long *sync_arr = NULL;                                                       \
        int shift;                                                                            \
        int to_nbr_idx, to_nbr;                                                               \
        int from_nbr_idx, from_nbr;                                                           \
        int temp = size - 1; /* used to calculate number of phases */                         \
        int myIdx = nvshmemi_thread_id_in_##SC();                                             \
        int groupSize = nvshmemi_##SC##_size();                                               \
        sync_arr = (volatile long *)pSync;                                                    \
        int pow_k = 1;                                                                        \
        int phase_num = 0;                                                                    \
        volatile long counter_val = NVSHMEMI_SYNC_VALUE + 1;                                  \
        volatile long *counter;                                                               \
        if (sync_counter == NULL)                                                             \
            counter = &counter_val;                                                           \
        else                                                                                  \
            counter = sync_counter;                                                           \
        while (temp) {                                                                        \
            /* notify neighbors */                                                            \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                             \
                shift = j << phase_num;                                                       \
                if (shift >= size) break;                                                     \
                                                                                              \
                to_nbr_idx = my_idx_in_active_set + shift;                                    \
                if (to_nbr_idx >= size) to_nbr_idx = to_nbr_idx - size;                       \
                to_nbr = start + to_nbr_idx * stride;                                         \
                                                                                              \
                nvshmemi_signal_for_barrier<long>(                                            \
                    ((long *)sync_arr + nvshmemi_device_state_d.mype), counter[0], to_nbr);   \
            }                                                                                 \
                                                                                              \
            /* wait for neighbors notification */                                             \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                             \
                shift = j << phase_num;                                                       \
                if (shift >= size) break;                                                     \
                                                                                              \
                from_nbr_idx = my_idx_in_active_set - shift;                                  \
                if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;                     \
                from_nbr = start + from_nbr_idx * stride;                                     \
                                                                                              \
                nvshmemi_wait_until_greater_than_equals<volatile long>(                       \
                    sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER_WARP);        \
                if (sync_counter == NULL) *(sync_arr + from_nbr) = NVSHMEMI_SYNC_VALUE;       \
            }                                                                                 \
            pow_k <<= logk;                                                                   \
            temp >>= logk;                                                                    \
            phase_num++;                                                                      \
            nvshmemi_##SC##_sync();                                                           \
        }                                                                                     \
        if (sync_counter) {                                                                   \
            if (!myIdx) sync_counter[0] += 1;                                                 \
            nvshmemi_##SC##_sync();                                                           \
        }                                                                                     \
    }

NVSHMEMI_SYNC_DISSEM_POW2_ALGO(thread, , )
NVSHMEMI_SYNC_DISSEM_POW2_ALGO(warp, _warp, x)
NVSHMEMI_SYNC_DISSEM_POW2_ALGO(block, _block, x)
#undef NVSHMEMI_SYNC_DISSEM_POW2_ALGO

#define NVSHMEMI_SYNC_DISSEM_ALGO(SC, SC_SUFFIX, SC_PREFIX)                                   \
    __device__ static inline void sync_dissem##SC_SUFFIX(                                     \
        int start, int stride, int size, volatile long *pSync, volatile long *sync_counter) { \
        int num_phases = 0;                                                                   \
        int k = min(nvshmemi_device_state_d.barrier_tg_dissem_kval, size); /* radix for the   \
                                                        dissemination algorithm */            \
        int my_idx_in_active_set = (nvshmemi_device_state_d.mype - start) / stride;           \
        volatile long *sync_arr = NULL;                                                       \
        int shift;                                                                            \
        int to_nbr_idx, to_nbr;                                                               \
        int from_nbr_idx, from_nbr;                                                           \
        int temp = size - 1; /* used to calculate number of phases */                         \
        while (temp) {                                                                        \
            num_phases++;                                                                     \
            temp /= k;                                                                        \
        }                                                                                     \
        int myIdx = nvshmemi_thread_id_in_##SC();                                             \
        int groupSize = nvshmemi_##SC##_size();                                               \
        sync_arr = (volatile long *)pSync;                                                    \
        int pow_k = 1;                                                                        \
        volatile long counter_val = NVSHMEMI_SYNC_VALUE + 1;                                  \
        volatile long *counter;                                                               \
        if (sync_counter == NULL)                                                             \
            counter = &counter_val;                                                           \
        else                                                                                  \
            counter = sync_counter;                                                           \
        for (int i = 0; i < num_phases; i++) {                                                \
            /* notify neighbors */                                                            \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                             \
                shift = j * pow_k;                                                            \
                if (shift >= size) break;                                                     \
                to_nbr_idx = (my_idx_in_active_set + shift) % size;                           \
                to_nbr = start + to_nbr_idx * stride;                                         \
                                                                                              \
                nvshmemi_signal_for_barrier<long>(                                            \
                    ((long *)sync_arr + nvshmemi_device_state_d.mype), counter[0], to_nbr);   \
            }                                                                                 \
                                                                                              \
            /* wait for neighbors notification */                                             \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                             \
                shift = j * pow_k;                                                            \
                if (shift >= size) break;                                                     \
                                                                                              \
                from_nbr_idx = my_idx_in_active_set - shift;                                  \
                if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;                     \
                from_nbr = start + from_nbr_idx * stride;                                     \
                                                                                              \
                nvshmemi_wait_until_greater_than_equals<volatile long>(                       \
                    sync_arr + from_nbr, counter[0], NVSHMEMI_CALL_SITE_BARRIER_WARP);        \
                if (sync_counter == NULL) *(sync_arr + from_nbr) = NVSHMEMI_SYNC_VALUE;       \
            }                                                                                 \
            pow_k *= k;                                                                       \
            nvshmemi_##SC##_sync();                                                           \
        }                                                                                     \
        if (sync_counter) {                                                                   \
            if (!myIdx) sync_counter[0] += 1;                                                 \
            nvshmemi_##SC##_sync();                                                           \
        }                                                                                     \
    }

NVSHMEMI_SYNC_DISSEM_ALGO(thread, , )
NVSHMEMI_SYNC_DISSEM_ALGO(warp, _warp, x)
NVSHMEMI_SYNC_DISSEM_ALGO(block, _block, x)
#undef NVSHMEMI_SYNC_DISSEM_ALGO

#define NVSHMEMI_BARRIER_THREADGROUP_ALGO(SC, SC_SUFFIX, SC_PREFIX)                      \
    __device__ static inline void nvshmemi_sync_algo##SC_SUFFIX(                         \
        int start, int stride, int size, volatile long *pSync, volatile long *counter) { \
        int k = min(nvshmemi_device_state_d.barrier_tg_dissem_kval, size);               \
        k = max(k, 2);                                                                   \
        switch (k) {                                                                     \
            case 2:                                                                      \
                sync_dissem_pow2##SC_SUFFIX<2, 1>(start, stride, size, pSync, counter);  \
                break;                                                                   \
            case 4:                                                                      \
                sync_dissem_pow2##SC_SUFFIX<4, 2>(start, stride, size, pSync, counter);  \
                break;                                                                   \
            case 8:                                                                      \
                sync_dissem_pow2##SC_SUFFIX<8, 3>(start, stride, size, pSync, counter);  \
                break;                                                                   \
            case 16:                                                                     \
                sync_dissem_pow2##SC_SUFFIX<16, 4>(start, stride, size, pSync, counter); \
                break;                                                                   \
            case 32:                                                                     \
                sync_dissem_pow2##SC_SUFFIX<32, 5>(start, stride, size, pSync, counter); \
                break;                                                                   \
            default:                                                                     \
                sync_dissem##SC_SUFFIX(start, stride, size, pSync, counter);             \
                break;                                                                   \
        }                                                                                \
    }
NVSHMEMI_BARRIER_THREADGROUP_ALGO(thread, , )
NVSHMEMI_BARRIER_THREADGROUP_ALGO(warp, _warp, x)
NVSHMEMI_BARRIER_THREADGROUP_ALGO(block, _block, x) 
#undef NVSHMEMI_SYNC_DISSEM_ALGO

#endif /* __CUDA_ARCH__ */

#endif /* _BARRIER_SYNC_ALGO_CUH_ */
