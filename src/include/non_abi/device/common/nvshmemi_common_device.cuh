/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef _NVSHMEM_COMMON_DEVICE_CUH_
#define _NVSHMEM_COMMON_DEVICE_CUH_

#include <cuda_runtime.h>
#if not defined __CUDACC_RTC__
#include <stdint.h>
#include <stddef.h>
#else
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#endif
#include "non_abi/nvshmem_build_options.h"
#include "device_host/nvshmem_common.cuh"
#include "device_host_transport/nvshmem_common_transport.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"

#ifdef __CUDA_ARCH__
__device__ int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe,
                                          nvshmemi_team_t *dest_team);
__device__ long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op);
__device__ long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team);

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_quiet() {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    if ((nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_LDST)) {
        nvshmemi_transfer_quiet<SCOPE>(true);
    } else {
        if (!myIdx)
            __threadfence_system(); /* Use __threadfence_system instead of __threadfence
                                     for data visibility in case of intra-node GPU transfers */
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_WARP>();
template __device__ void nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>();

__device__ inline void nvshmemi_fence() {
    if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_LDST) {
        nvshmemi_transfer_fence<NVSHMEMI_THREADGROUP_THREAD>();
    }
    __threadfence_system(); /* Use __threadfence_system instead of __threadfence
                               for data visibility in case of intra-node GPU transfers */
}

template <typename T>
__device__ inline int nvshmemi_test(volatile T *ivar, int cmp, T cmp_value) {
    int return_value = 0;
    if (NVSHMEM_CMP_GE == cmp) {
        if (*ivar >= cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_EQ == cmp) {
        if (*ivar == cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_NE == cmp) {
        if (*ivar != cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_GT == cmp) {
        if (*ivar > cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_LT == cmp) {
        if (*ivar < cmp_value) return_value = 1;
    } else if (NVSHMEM_CMP_LE == cmp) {
        if (*ivar <= cmp_value) return_value = 1;
    }
    return return_value;
}

template <threadgroup_t SCOPE>
__device__ inline void nvshmemi_memcpy_threadgroup(void *__restrict__ dst,
                                                   const void *__restrict__ src, size_t len) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    /*
     * If src and dst are 16B aligned copy as much as possible using 16B chunks
     */
    if ((uintptr_t)dst % 16 == 0 && (uintptr_t)src % 16 == 0) {
        int4 *__restrict__ dst_p = (int4 *)dst;
        const int4 *__restrict__ src_p = (const int4 *)src;
        const size_t nelems = len / 16;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 16;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 8B aligned copy as much as possible using 8B chunks
     */
    if ((uintptr_t)dst % 8 == 0 && (uintptr_t)src % 8 == 0) {
        uint64_t *__restrict__ dst_p = (uint64_t *)dst;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        const size_t nelems = len / 8;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 8;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 4B aligned copy as much as possible using 4B chunks
     */
    if ((uintptr_t)dst % 4 == 0 && (uintptr_t)src % 4 == 0) {
        uint32_t *__restrict__ dst_p = (uint32_t *)dst;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        const size_t nelems = len / 4;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 4;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    /*
     * If src and dst are 2B aligned copy as much as possible using 2B chunks
     */
    if ((uintptr_t)dst % 2 == 0 && (uintptr_t)src % 2 == 0) {
        uint16_t *__restrict__ dst_p = (uint16_t *)dst;
        const uint16_t *__restrict__ src_p = (const uint16_t *)src;
        const size_t nelems = len / 2;

        for (size_t i = myIdx; i < nelems; i += groupSize) dst_p[i] = src_p[i];

        len -= nelems * 2;

        if (0 == len) return;

        dst = (void *)(dst_p + nelems);
        src = (void *)(src_p + nelems);
    }

    unsigned char *__restrict__ dst_c = (unsigned char *)dst;
    const unsigned char *__restrict__ src_c = (const unsigned char *)src;

    for (size_t i = myIdx; i < len; i += groupSize) dst_c[i] = src_c[i];
}

template <typename T>
__device__ inline void nvshmemi_p(T *dest, const T value, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        T *dest_actual = (T *)((char *)(peer_base_addr) +
                               ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = value;
    } else {
        nvshmemi_transfer_rma_p<T>((void *)dest, value, pe);
    }
}

template <typename T>
__device__ inline T nvshmemi_g(const T *source, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        T *source_actual = (T *)((char *)(peer_base_addr) +
                                 ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));
        return *source_actual;
    } else {
        return nvshmemi_transfer_rma_g<T>((void *)source, pe);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_put_threadgroup(T *dest, const T *source, size_t nelems, int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_PUT>((void *)dest, (void *)source,
                                                      nelems * sizeof(T), pe);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ inline void nvshmemi_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (sig_op == NVSHMEMI_AMO_SIGNAL_SET && peer_base_addr != NULL) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = signal;
    } else if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        /* sig_op == NVSHMEM_SIGNAL_ADD */
        atomicAdd_system((unsigned long long *)dest_actual, signal);
    } else {
        nvshmemi_transfer_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe,
                                                 (nvshmemi_amo_t)sig_op);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemii_put_signal_threadgroup(T *dest, const T *source, size_t nelems,
                                                        uint64_t *sig_addr, uint64_t signal,
                                                        int sig_op, int pe, bool is_nbi) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source,
                                           nelems * sizeof(T));
        nvshmemi_threadgroup_sync<SCOPE>();
        if (!myIdx) {
            __threadfence_system();
            nvshmemi_signal_op(sig_addr, signal, sig_op, pe);
        }
    } else {
        nvshmemi_transfer_put_signal<SCOPE>((void *)dest, (void *)source, nelems * sizeof(T),
                                            (void *)sig_addr, signal, (nvshmemi_amo_t)sig_op, pe,
                                            is_nbi);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_put_signal_threadgroup(T *dest, const T *source, size_t nelems,
                                                       uint64_t *sig_addr, uint64_t signal,
                                                       int sig_op, int pe, bool is_nbi) {
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemii_put_signal_threadgroup<T, SCOPE>(dest, source, nelems, sig_addr, signal, sig_op, pe,
                                               is_nbi);
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_get_threadgroup(T *dest, const T *source, size_t nelems, int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *source_actual = (char *)(peer_base_addr) +
                              ((char *)source - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest, (const void *)source_actual,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_GET>((void *)source, (void *)dest,
                                                      nelems * sizeof(T), pe);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemii_put_nbi_threadgroup(T *dest, const T *source, size_t nelems,
                                                     int pe) {
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *dest_actual =
            (char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest_actual, (const void *)source,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_PUT>((void *)dest, (void *)source,
                                                          nelems * sizeof(T), pe);
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_put_nbi_threadgroup(T *dest, const T *source, size_t nelems,
                                                    int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    nvshmemii_put_nbi_threadgroup<T, SCOPE>(dest, source, nelems, pe);
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_get_nbi_threadgroup(T *dest, const T *source, size_t nelems,
                                                    int pe) {
    nvshmemi_threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *source_actual = (char *)(peer_base_addr) +
                              ((char *)source - (char *)(nvshmemi_device_state_d.heap_base));
        nvshmemi_memcpy_threadgroup<SCOPE>((void *)dest, (const void *)source_actual,
                                           nelems * sizeof(T));
    } else {
        nvshmemi_transfer_rma_nbi<SCOPE, NVSHMEMI_OP_GET>((void *)source, (void *)dest,
                                                          nelems * sizeof(T), pe);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_recvLL(T *dest, const uint64_t *src, size_t nelems, uint32_t flag) {
    // Assumptions: sizeof(T) >= 4 bytes, num_subelems is a multiple of 2
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));

    uint32_t flag1, flag2, data1, data2;
    for (int i = 2 * myIdx; i < num_subelems; i += 2 * groupSize) {
        do {
            asm("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                : "l"(&src[i]));
        } while ((flag1 != flag) || (flag2 != flag));
        // printf("received: %d %d\n", data1, data2);
        *(uint32_t *)((char *)dest + i * sizeof(uint32_t)) = data1;
        *(uint32_t *)((char *)dest + (i + 1) * sizeof(uint32_t)) = data2;
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ inline void nvshmemi_packLL(uint64_t *dest, const T *source, size_t nelems,
                                       uint32_t ll_flag) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    size_t num_subelems = nelems * (sizeof(T) / sizeof(uint32_t));
    for (int i = myIdx; i < num_subelems; i += groupSize) {
        size_t dst_offset = 2 * i * sizeof(uint32_t);
        size_t src_offset = i * sizeof(uint32_t);
        *(uint32_t *)((char *)dest + dst_offset) = *(uint32_t *)((char *)source + src_offset);
        *(uint32_t *)((char *)dest + dst_offset + sizeof(uint32_t)) = ll_flag;
    }
}

__device__ inline void *nvshmemi_ptr(const void *ptr, int pe) {
    ptrdiff_t offset = (char *)ptr - (char *)nvshmemi_device_state_d.heap_base;

    if (ptr >= nvshmemi_device_state_d.heap_base && offset < nvshmemi_device_state_d.heap_size) {
        void *peer_addr = (void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
        if (peer_addr != NULL) peer_addr = (void *)((char *)peer_addr + offset);
        return peer_addr;
    } else
        return NULL;
}

#endif /* __CUDA__ARCH__ */
#endif /* _NVSHMEM_COMMON_DEVICE_CUH_ */
