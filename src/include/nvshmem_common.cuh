/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEM_COMMON_H_
#define _NVSHMEM_COMMON_H_

#include <cuda_runtime.h>
#ifdef NVSHMEM_COMPLEX_SUPPORT
#include <complex.h>
#endif
#include "nvshmem_constants.h"
#include "nvshmemi_constants.h"

#ifdef __CUDA_ARCH__
#  ifdef NVSHMEMI_HOST_ONLY
#    define NVSHMEMI_HOSTDEVICE_PREFIX __host__
#  else
#    define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__
#  endif
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX
#endif

#define NVSHMEMI_UNUSED_ARG(ARG) (void)(ARG)

typedef int nvshmem_team_t;

#define NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(OPGRPNAME, opname)                \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint, unsigned int, opname)                  \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulong, unsigned long, opname)                \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulonglong, unsigned long long, opname)       \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int32, int32_t, opname)                      \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint32, uint32_t, opname)                    \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int64, int64_t, opname)                      \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint64, uint64_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(OPGRPNAME, opname)               \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int, int, opname)                            \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(long, long, opname)                          \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(longlong, long long, opname)                 \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(size, size_t, opname)                        \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ptrdiff, ptrdiff_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(OPGRPNAME, opname)               \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(float, float, opname)                        \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(double, double, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_FN_TEMPLATE)      \
    NVSHMEMI_FN_TEMPLATE(float, float)                                  \
    NVSHMEMI_FN_TEMPLATE(double, double)                                \
    NVSHMEMI_FN_TEMPLATE(char, char)                                    \
    NVSHMEMI_FN_TEMPLATE(short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(schar, signed char)                            \
    NVSHMEMI_FN_TEMPLATE(int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(uchar, unsigned char)                          \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(int8, int8_t)                                  \
    NVSHMEMI_FN_TEMPLATE(int16, int16_t)                                \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint8, uint8_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint16, uint16_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_FN_TEMPLATE)                   \
    NVSHMEMI_FN_TEMPLATE(8)                                             \
    NVSHMEMI_FN_TEMPLATE(16)                                            \
    NVSHMEMI_FN_TEMPLATE(32)                                            \
    NVSHMEMI_FN_TEMPLATE(64)                                            \
    NVSHMEMI_FN_TEMPLATE(128)

#define NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMI_FN_TEMPLATE)         \
    NVSHMEMI_FN_TEMPLATE(8, int8_t)                                     \
    NVSHMEMI_FN_TEMPLATE(16, int16_t)                                   \
    NVSHMEMI_FN_TEMPLATE(32, int32_t)                                   \
    NVSHMEMI_FN_TEMPLATE(64, int64_t)                                   \
    NVSHMEMI_FN_TEMPLATE(128, int4)


#define NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_FN_TEMPLATE)              \
    NVSHMEMI_FN_TEMPLATE(short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMX_REPT_FOR_SIGNAL_TYPES(NVSHMEMI_FN_TEMPLATE)            \
    NVSHMEMI_FN_TEMPLATE(short, short)                                  \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                        \
    NVSHMEMI_FN_TEMPLATE(int, int)                                      \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                            \
    NVSHMEMI_FN_TEMPLATE(long, long)                                    \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                          \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                           \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)                 \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                                \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                                \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                              \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                              \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                                  \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_FN_TEMPLATE(short, short, opname)                          \
    NVSHMEMI_FN_TEMPLATE(int, int, opname)                              \
    NVSHMEMI_FN_TEMPLATE(long, long, opname)                            \
    NVSHMEMI_FN_TEMPLATE(longlong, long, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_FN_TEMPLATE(short, short, opname)                          \
    NVSHMEMI_FN_TEMPLATE(int, int, opname)                              \
    NVSHMEMI_FN_TEMPLATE(long, long, opname)                            \
    NVSHMEMI_FN_TEMPLATE(longlong, long, opname)                        \
    NVSHMEMI_FN_TEMPLATE(float, float, opname)                          \
    NVSHMEMI_FN_TEMPLATE(double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)  \
    NVSHMEMI_FN_TEMPLATE(complexf, double complex, opname)              \
    NVSHMEMI_FN_TEMPLATE(complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)
#endif

#define NVSHMEMI_DECL_THREAD_IDX_warp()                         \
    ;                                                           \
    int myIdx;                                                  \
    asm volatile("mov.u32  %0, %laneid;" : "=r"(myIdx));

#define NVSHMEMI_DECL_THREADGROUP_SIZE_warp()                           \
    ;                                                                   \
    int groupSize = ((blockDim.x * blockDim.y * blockDim.z) < warpSize) \
                        ? (blockDim.x * blockDim.y * blockDim.z)        \
                        : warpSize;

#define NVSHMEMI_DECL_THREAD_IDX_block()                        \
    ;                                                           \
    int myIdx = (threadIdx.x + threadIdx.y * blockDim.x +       \
                 threadIdx.z * blockDim.x * blockDim.y);

#define NVSHMEMI_DECL_THREADGROUP_SIZE_block()                  \
    ;                                                           \
    int groupSize = (blockDim.x * blockDim.y * blockDim.z);


#define NVSHMEMI_SYNC_warp() \
    ;                        \
    __syncwarp();

#define NVSHMEMI_SYNC_block() \
    ;                         \
    __syncthreads();

#define NVSHMEMI_SYNC_thread() ;

extern __constant__ int nvshmemi_mype_d;
extern __constant__ int nvshmemi_npes_d;
extern __constant__ int nvshmemi_node_mype_d;
extern __constant__ int nvshmemi_node_npes_d;
extern __constant__ int *nvshmemi_p2p_attrib_native_atomic_support_d;
extern __constant__ int nvshmemi_proxy_d;
extern __constant__ int nvshmemi_atomics_sync_d;
extern __constant__ int nvshmemi_job_connectivity_d;
extern __constant__ void *nvshmemi_heap_base_d;
extern __constant__ size_t nvshmemi_heap_size_d;
extern __constant__ void **nvshmemi_peer_heap_base_d;

enum nvshmemi_call_site_id {
    NVSHMEMI_CALL_SITE_BARRIER = 0,
    NVSHMEMI_CALL_SITE_BARRIER_WARP,
    NVSHMEMI_CALL_SITE_BARRIER_THREADBLOCK,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_GE,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_EQ,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_NE,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_GT,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_LT,
    NVSHMEMI_CALL_SITE_WAIT_UNTIL_LE,
    NVSHMEMI_CALL_SITE_WAIT_NE,
    NVSHMEMI_CALL_SITE_PROXY_CHECK_CHANNEL_AVAILABILITY,
    NVSHMEMI_CALL_SITE_PROXY_QUIET,
    NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET,
};

#define TIMEOUT_NCYCLES 1e9

typedef struct {
    uint64_t signal;
    uint64_t caller;
    uint64_t signal_addr;
    uint64_t signal_val_found;
    uint64_t signal_val_expected;
} nvshmemi_timeout_t;

extern __device__ nvshmemi_timeout_t *nvshmemi_timeout_d;
extern __device__ unsigned long long test_wait_any_start_idx_d;
__device__ void nvshmemi_proxy_enforce_consistency_at_target();

template <typename T>
__device__ inline void nvshmemi_check_timeout_and_log(long long int start, int caller, uintptr_t signal_addr,
                                             T signal_val_found, T signal_val_expected) {
    long long int now;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(now));
    if ((now - start) > TIMEOUT_NCYCLES) {
        nvshmemi_timeout_t *timeout_d = nvshmemi_timeout_d;
        timeout_d->caller = caller;
        timeout_d->signal_addr = signal_addr;
        *(T *)(&timeout_d->signal_val_found) = signal_val_found;
        *(T *)(&timeout_d->signal_val_expected) = signal_val_expected;
        *((volatile uint64_t *)(&timeout_d->signal)) = 1;
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_greater_than(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr <= val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_greater_than_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr < val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_lesser_than(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr >= val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_lesser_than_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr > val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr != val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_not_equals(volatile T *addr, T val, int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    while (*addr == val) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, *addr, val);
#endif
    }
}

template <typename T>
__device__ inline void nvshmemi_wait_until_greater_than_equals_add(volatile T *addr, uint64_t toadd, T val,
                                                          int caller) {
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    long long int start;
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
#endif
    T valataddr;
    do {
        valataddr = *addr;
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        nvshmemi_check_timeout_and_log<T>(start, caller, (uintptr_t)addr, valataddr + toadd, val);
#endif
    } while (valataddr + toadd < val);
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

template <typename T>
__device__ inline void nvshmemi_wait_until(volatile T *ivar, int cmp, T cmp_value) {
    if (NVSHMEM_CMP_GE == cmp) {
        nvshmemi_wait_until_greater_than_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_GE);
    } else if (NVSHMEM_CMP_EQ == cmp) {
        nvshmemi_wait_until_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_EQ);
    } else if (NVSHMEM_CMP_NE == cmp) {
        nvshmemi_wait_until_not_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_NE);
    } else if (NVSHMEM_CMP_GT == cmp) {
        nvshmemi_wait_until_greater_than<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_GT);
    } else if (NVSHMEM_CMP_LT == cmp) {
        nvshmemi_wait_until_lesser_than<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_LT);
    } else if (NVSHMEM_CMP_LE == cmp) {
        nvshmemi_wait_until_lesser_than_equals<T>(ivar, cmp_value, NVSHMEMI_CALL_SITE_WAIT_UNTIL_LE);
    }
}

#ifdef __CUDA_ARCH__
__device__ inline void nvshmemi_syncapi_update_mem() {
    __threadfence(); /* 1. Ensures consitency op is not called before the prior test/wait condition has been met
                        2. Needed to prevent reorder of instructions after sync api (when the following if condition is false) */
    if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY) { 
       nvshmemi_proxy_enforce_consistency_at_target();
    }	
}
#endif
#endif
