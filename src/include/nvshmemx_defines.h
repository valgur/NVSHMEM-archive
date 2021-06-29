/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_DEFINES_H_
#define _NVSHMEMX_DEFINES_H_

#include "nvshmemi_util.h"
#include "nvshmem_common.cuh"

#ifdef __CUDA_ARCH__
template <typename T>
__device__ inline void nvshmemi_signal(T *dest, const T value, int pe) {
   const void *peer_base_addr =
       (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);
   if (peer_base_addr != NULL) {
       volatile T *dest_actual = (volatile T *)((char *)(peer_base_addr) +
                              ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
       *dest_actual = value;
   } else {
       nvshmemi_proxy_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
   }
}

#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)     \
    __device__ inline void nvshmemi_##TYPENAME##_put_signal##SC_SUFFIX(                     \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, \
        int sig_op, int pe, bool is_nbi) {                                                  \
        NVSHMEMI_DECL_THREAD_IDX##SC_SUFFIX();                                              \
        void *peer_base_addr = (void *)__ldg(                                               \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);       \
        if (peer_base_addr) {                                                               \
            nvshmemx_##TYPENAME##_put##SC_SUFFIX(dest, source, nelems, pe);                 \
            if (myIdx == 0) {                                                               \
                __threadfence_system();                                                     \
                nvshmemx_signal_op(sig_addr, signal, sig_op, pe);                           \
            }                                                                               \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                     \
        } else {                                                                            \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                     \
            if (myIdx == 0) {                                                               \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,       \
                                                        nelems * sizeof(TYPE), pe);         \
                nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe,         \
                                                      (nvshmemi_amo_t)sig_op);              \
                if (is_nbi == 0) nvshmemi_proxy_quiet(true);                                \
            }                                                                               \
            NVSHMEMI_SYNC##SC_SUFFIX();                                                     \
        }                                                                                   \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE, block, _block,
                                                 x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDA_ARCH__

#define NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_put_##Group(Type *dest, const Type *source, \
                                                         size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                         \
        void *peer_base_addr =                                                           \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);    \
        if (peer_base_addr) {                                                            \
            Type *dest_actual = (Type *)((char *)(peer_base_addr) +                      \
                                 ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));       \
            nvshmemi_memcpy_##Group(dest_actual, source, nelems*sizeof(Type));           \
        } else {                                                                         \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                          \
            if (!myIdx) {                                                                \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,         \
                                                   nelems * sizeof(Type), pe);           \
                nvshmemi_proxy_quiet(true);                                              \
            }                                                                            \
        }                                                                                \
        NVSHMEMI_SYNC_##Group();                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP

/* __device__ nvshmem_<typename>_put_signal_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ inline void nvshmemx_##TYPENAME##_put_signal##SC_SUFFIX(                      \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,  \
        int sig_op, int pe) {                                                                \
        nvshmemi_##TYPENAME##_put_signal##SC_SUFFIX(dest, source, nelems, sig_addr, signal,  \
                                                    sig_op, pe, 0);                          \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_SCOPE_IMPL

/* __device__ nvshmem_putmem_signal_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)                            \
    __device__ inline void nvshmemx_putmem_signal##SC_SUFFIX(                                     \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,       \
        int sig_op, int pe) {                                                                     \
        nvshmemi_char_put_signal##SC_SUFFIX((char *)dest, (const char *)source, nelems, sig_addr, \
                                            signal, sig_op, pe, 0);                               \
    }

NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_SCOPE_IMPL

/* __device__ nvshmem_putsize_signal_scope */
#define NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)                 \
    __device__ inline void nvshmemx_put##BITS##_signal##SC_SUFFIX(                            \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,   \
        int sig_op, int pe) {                                                                 \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems *(BITS / 8), sig_addr, signal, \
                                          sig_op, pe);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_SCOPE_IMPL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_FOR_SCOPE

/* __device__ nvshmem_<typename>_put_signal_nbi_scope */
#define NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ inline void nvshmemx_##TYPENAME##_put_signal_nbi##SC_SUFFIX(                      \
        TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,      \
        int sig_op, int pe) {                                                                    \
        nvshmemi_##TYPENAME##_put_signal##SC_SUFFIX(dest, source, nelems, sig_addr, signal,      \
                                                    sig_op, pe, 1);                              \
    }
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, warp,
                                                 _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL, block,
                                                 _block, x)
#undef NVSHMEMI_TYPENAME_PUT_SIGNAL_NBI_SCOPE_IMPL

/* __device__ nvshmem_putmem_signal_nbi_scope */
#define NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)                        \
    __device__ inline void nvshmemx_putmem_signal_nbi##SC_SUFFIX(                                 \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,       \
        int sig_op, int pe) {                                                                     \
        nvshmemi_char_put_signal##SC_SUFFIX((char *)dest, (const char *)source, nelems, sig_addr, \
                                            signal, sig_op, pe, 1);                               \
    }

NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(warp, _warp, x)
NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL(block, _block, x)
#undef NVSHMEMI_PUTMEM_SIGNAL_NBI_SCOPE_IMPL

#define NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX, BITS)             \
    __device__ inline void nvshmemx_put##BITS##_signal_nbi##SC_SUFFIX(                        \
        void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal,   \
        int sig_op, int pe) {                                                                 \
        nvshmemx_putmem_signal##SC_SUFFIX(dest, source, nelems *(BITS / 8), sig_addr, signal, \
                                          sig_op, pe);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL, warp, _warp, x)
NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_PUTSIZE_SIGNAL_NBI_SCOPE_IMPL, block, _block, x)
#undef NVSHMEMI_REPT_PUTSIZE_SIGNAL_NBI_FOR_SCOPE

#define NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_get_##Group(Type *dest, const Type *source, \
                                                         size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                         \
        void *peer_base_addr =                                                           \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);    \
        if (peer_base_addr) {                                                            \
            Type *source_actual = (Type *)((char *)(peer_base_addr) +                    \
                                   ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));   \
            nvshmemi_memcpy_##Group(dest, source_actual, nelems*sizeof(Type));           \
        } else {                                                                         \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                          \
            if (!myIdx) {                                                                \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,         \
                                                   nelems * sizeof(Type), pe);           \
                nvshmemi_proxy_quiet(true);                                              \
            }                                                                            \
        }                                                                                \
        NVSHMEMI_SYNC_##Group();                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_GET(Name, Type)        \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET)
#undef DEFINE_NVSHMEM_TYPE_GET

#define NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, Group)                                           \
    __device__ inline void nvshmemx_put##Name##_##Group(void *dest, const void *source,          \
                                                        size_t nelems, int pe) {                 \
        NVSHMEMI_SYNC_##Group();                                                                 \
        void *peer_base_addr = (void *)__ldg(                                                    \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);            \
        if (peer_base_addr) {                                                                    \
            Type *dest_actual =                                                                  \
                (Type *)((char *)(peer_base_addr) +                                              \
                                  ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base))); \
            nvshmemi_memcpy_##Group(dest_actual, source, nelems * sizeof(Type));                 \
        } else {                                                                                 \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                  \
            if (!myIdx) {                                                                        \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,            \
                                                        nelems * sizeof(Type), pe);              \
                nvshmemi_proxy_quiet(true);                                                      \
            }                                                                                    \
        }                                                                                        \
        NVSHMEMI_SYNC_##Group();                                                                 \
    }

#define DEFINE_NVSHMEM_PUTSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_THREADGROUP


#define NVSHMEM_GETSIZE_THREADGROUP(Name, Type, Group)                                         \
    __device__ inline void nvshmemx_get##Name##_##Group(void *dest, const void *source,        \
                                                        size_t nelems, int pe) {               \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr =                                                                 \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);          \
        if (peer_base_addr) {                                                                  \
            char *source_actual = ((char *)(peer_base_addr) +                                  \
                                   ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));         \
            nvshmemi_memcpy_##Group(dest, source_actual, nelems*sizeof(Type));                 \
        } else {                                                                               \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                \
            if (!myIdx) {                                                                      \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,               \
                                                   nelems * sizeof(Type), pe);                 \
                nvshmemi_proxy_quiet(true);                                                    \
            }                                                                                  \
        }                                                                                      \
        NVSHMEMI_SYNC_##Group();                                                               \
    }

#define DEFINE_NVSHMEM_GETSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_THREADGROUP(Group)                                                 \
    __device__ inline void nvshmemx_putmem_##Group(void *dest, const void *source, size_t bytes, \
                                                   int pe) {                                     \
        NVSHMEMI_SYNC_##Group();                                                                 \
        void *peer_base_addr =                                                                   \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);            \
        if (peer_base_addr) {                                                                    \
            char *dest_actual = ((char *)(peer_base_addr) +                                      \
                                 ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));               \
            nvshmemi_memcpy_##Group(dest_actual, source, bytes);                                 \
        } else {                                                                                 \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                  \
            if (!myIdx) {                                                                        \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, bytes, pe);     \
                nvshmemi_proxy_quiet(true);                                                      \
            }                                                                                    \
        }                                                                                        \
        NVSHMEMI_SYNC_##Group();                                                                 \
    }

DEFINE_NVSHMEM_PUTMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_THREADGROUP(Group)                                                  \
    __device__ inline void nvshmemx_getmem_##Group(void *dest, const void *source, size_t bytes,  \
                                                   int pe) {                                      \
        NVSHMEMI_SYNC_##Group();                                                                  \
        void *peer_base_addr = (void *)__ldg(                                                     \
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);             \
        if (peer_base_addr) {                                                                     \
            char *source_actual =                                                                 \
                ((char *)(peer_base_addr) +                                                       \
                 ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));                 \
            nvshmemi_memcpy_##Group(dest, source_actual, bytes);                                  \
        } else {                                                                                  \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                   \
            if (!myIdx) {                                                                         \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest, bytes, pe); \
                nvshmemi_proxy_quiet(true);                                                       \
            }                                                                                     \
        }                                                                                         \
        NVSHMEMI_SYNC_##Group();                                                                  \
    }

DEFINE_NVSHMEM_GETMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_THREADGROUP(block)

#define NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_put_nbi_##Group(Type *dest, const Type *source, \
                                                             size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                             \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);        \
        if (peer_base_addr) {                                                                \
            Type *dest_actual = (Type *)((char *)(peer_base_addr) +                          \
                                 ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));           \
            nvshmemi_memcpy_##Group(dest_actual, source, nelems*sizeof(Type));               \
        } else {                                                                             \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                              \
            if (!myIdx) {                                                                    \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,             \
                                                   nelems * sizeof(Type), pe);               \
            }                                                                                \
        }                                                                                    \
        NVSHMEMI_SYNC_##Group();                                                             \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP

#define NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_get_nbi_##Group(Type *dest, const Type *source, \
                                                             size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                             \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);        \
        if (peer_base_addr) {                                                                \
            Type *source_actual = (Type *)((char *)(peer_base_addr) +                        \
                                   ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));       \
            nvshmemi_memcpy_##Group(dest, source_actual, nelems*sizeof(Type));               \
        } else {                                                                             \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                              \
            if (!myIdx) {                                                                    \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,             \
                                                   nelems * sizeof(Type), pe);               \
            }                                                                                \
        }                                                                                    \
        NVSHMEMI_SYNC_##Group();                                                             \
    }

#define DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP

#define NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_put##Name##_nbi_##Group(void *dest, const void *source, \
                                                            size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                            \
        void *peer_base_addr =                                                              \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);       \
        if (peer_base_addr) {                                                               \
            Type *dest_actual = (Type *)((char *)(peer_base_addr) +                         \
                                  ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));         \
            nvshmemi_memcpy_##Group(dest_actual, source, nelems*sizeof(Type));              \
        } else {                                                                            \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                             \
            if (!myIdx) {                                                                   \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,            \
                                                        nelems * sizeof(Type), pe);              \
            }                                                                                    \
        }                                                                                        \
        NVSHMEMI_SYNC_##Group();                                                                 \
    }

#define DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP

#define NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_get##Name##_nbi_##Group(void *dest, const void *source, \
                                                            size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                            \
        void *peer_base_addr =                                                              \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);       \
        if (peer_base_addr) {                                                               \
            Type *source_actual = (Type *)((char *)(peer_base_addr) +                       \
                                   ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));      \
            nvshmemi_memcpy_##Group(dest, source_actual, nelems*sizeof(Type));              \
        } else {                                                                            \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                             \
            if (!myIdx) {                                                                   \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,            \
                                                   nelems * sizeof(Type), pe);              \
            }                                                                               \
        }                                                                                   \
        NVSHMEMI_SYNC_##Group();                                                            \
    }

#define DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(Group)                                         \
    __device__ inline void nvshmemx_putmem_nbi_##Group(void *dest, const void *source,       \
                                                       size_t bytes, int pe) {               \
        NVSHMEMI_SYNC_##Group();                                                             \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);        \
        if (peer_base_addr) {                                                                \
            char *dest_actual = (char *)((char *)(peer_base_addr) +                          \
                                 ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));           \
            nvshmemi_memcpy_##Group(dest_actual, source, bytes);                             \
        } else {                                                                             \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                              \
            if (!myIdx) {                                                                    \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, bytes, pe); \
            }                                                                                     \
        }                                                                                         \
        NVSHMEMI_SYNC_##Group();                                                                  \
    }

DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(Group)                                           \
    __device__ inline void nvshmemx_getmem_nbi_##Group(void *dest, const void *source,         \
                                                       size_t bytes, int pe) {                 \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr =                                                                 \
            (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);          \
        if (peer_base_addr) {                                                                  \
            char *source_actual = ((char *)(peer_base_addr) +                                  \
                                   ((char *)source - (char *)(nvshmemi_device_state_d.heap_base)));         \
            nvshmemi_memcpy_##Group(dest, source_actual, bytes);                               \
        } else {                                                                               \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                \
            if (!myIdx) {                                                                      \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest, bytes, pe);   \
            }                                                                                  \
        }                                                                                      \
        NVSHMEMI_SYNC_##Group();                                                               \
    }

DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(block)

/* __device__ nvshmem_signal */
#define DEFINE_NVSHMEMX_TYPE_SIGNAL(TYPENAME, TYPE)                                             \
    __device__ inline void nvshmemx_##TYPENAME##_signal(TYPE *dest, const TYPE value, int pe) { \
        nvshmemi_signal<TYPE>(dest, value, pe);                                                 \
}

NVSHMEMX_REPT_FOR_SIGNAL_TYPES(DEFINE_NVSHMEMX_TYPE_SIGNAL)
#undef DEFINE_NVSHMEMX_TYPE_SIGNAL

__device__ inline void nvshmemi_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base + pe);
    if (sig_op == NVSHMEM_SIGNAL_SET && peer_base_addr != NULL) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        *dest_actual = signal;
    } else if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {
        volatile uint64_t *dest_actual =
            (volatile uint64_t *)((char *)(peer_base_addr) +
                                  ((char *)sig_addr - (char *)(nvshmemi_device_state_d.heap_base)));
        /* sig_op == NVSHMEM_SIGNAL_ADD */
        atomicAdd((unsigned long long *)dest_actual, signal);
    } else {
        nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
    }
}

__device__ inline void nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmemi_signal_op(sig_addr, signal, sig_op, pe);
}

#endif /* __CUDA_ARCH__ */

#ifdef __cplusplus
}
#endif
#endif
