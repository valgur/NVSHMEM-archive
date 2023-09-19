/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
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

#if not defined __CUDACC_RTC__
#include <stdint.h>
#else
#include "cuda/std/cstdint"
#if not defined SIZE_MAX
#define SIZE_MAX (1ULL << 63)
#endif
#endif
#include <cuda_runtime.h>
#ifdef NVSHMEM_COMPLEX_SUPPORT
#include <complex.h>
#endif
#include "common/nvshmem_build_options.h"
#include "common/nvshmem_common_transport.h"
#include "common/nvshmem_types.h"
#include "common/nvshmem_constants.h"

#ifdef __CUDA_ARCH__
#ifdef NVSHMEMI_HOST_ONLY
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__
#else
#ifdef NVSHMEMI_DEVICE_ONLY
#define NVSHMEMI_HOSTDEVICE_PREFIX __device__
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX __host__ __device__
#endif
#endif
#else
#define NVSHMEMI_HOSTDEVICE_PREFIX
#endif

#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_DEVICE_INLINE inline
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#endif

#define NVSHPRI_float "%0.2f"
#define NVSHPRI_double "%0.2f"
#define NVSHPRI_char "%hhd"
#define NVSHPRI_schar "%hhd"
#define NVSHPRI_short "%hd"
#define NVSHPRI_int "%d"
#define NVSHPRI_long "%ld"
#define NVSHPRI_longlong "%lld"
#define NVSHPRI_uchar "%hhu"
#define NVSHPRI_ushort "%hu"
#define NVSHPRI_uint "%u"
#define NVSHPRI_ulong "%lu"
#define NVSHPRI_ulonglong "%llu"
#define NVSHPRI_int8 "%" PRIi8
#define NVSHPRI_int16 "%" PRIi16
#define NVSHPRI_int32 "%" PRIi32
#define NVSHPRI_int64 "%" PRIi64
#define NVSHPRI_uint8 "%" PRIu8
#define NVSHPRI_uint16 "%" PRIu16
#define NVSHPRI_uint32 "%" PRIu32
#define NVSHPRI_uint64 "%" PRIu64
#define NVSHPRI_size "%zu"
#define NVSHPRI_ptrdiff "%zu"
#define NVSHPRI_bool "%s"
#define NVSHPRI_string "\"%s\""

#define NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(OPGRPNAME, opname)                  \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint, unsigned int, opname)                    \
        NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulong, unsigned long, opname)              \
            NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulonglong, unsigned long long, opname) \
                NVSHMEMI_DECL_TYPE_##OPGRPNAME(int32, int32_t, opname)            \
                    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint32, uint32_t, opname)      \
                        NVSHMEMI_DECL_TYPE_##OPGRPNAME(int64, int64_t, opname)    \
                            NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint64, uint64_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(OPGRPNAME, opname)       \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int, int, opname)                    \
        NVSHMEMI_DECL_TYPE_##OPGRPNAME(long, long, opname)              \
            NVSHMEMI_DECL_TYPE_##OPGRPNAME(longlong, long long, opname) \
                NVSHMEMI_DECL_TYPE_##OPGRPNAME(size, size_t, opname)    \
                    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ptrdiff, ptrdiff_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(OPGRPNAME, opname) \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(float, float, opname)          \
        NVSHMEMI_DECL_TYPE_##OPGRPNAME(double, double, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(float, float)                             \
    NVSHMEMI_FN_TEMPLATE(double, double)                           \
    NVSHMEMI_FN_TEMPLATE(char, char)                               \
    NVSHMEMI_FN_TEMPLATE(short, short)                             \
    NVSHMEMI_FN_TEMPLATE(schar, signed char)                       \
    NVSHMEMI_FN_TEMPLATE(int, int)                                 \
    NVSHMEMI_FN_TEMPLATE(long, long)                               \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                      \
    NVSHMEMI_FN_TEMPLATE(uchar, unsigned char)                     \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)                   \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                       \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)                     \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)            \
    NVSHMEMI_FN_TEMPLATE(int8, int8_t)                             \
    NVSHMEMI_FN_TEMPLATE(int16, int16_t)                           \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                           \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                           \
    NVSHMEMI_FN_TEMPLATE(uint8, uint8_t)                           \
    NVSHMEMI_FN_TEMPLATE(uint16, uint16_t)                         \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                         \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                         \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                             \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC) \
    NVSHMEMI_FN_TEMPLATE(SC, float, float)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, double, double)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, char, char)                                          \
    NVSHMEMI_FN_TEMPLATE(SC, short, short)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, schar, signed char)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, int, int)                                            \
    NVSHMEMI_FN_TEMPLATE(SC, long, long)                                          \
    NVSHMEMI_FN_TEMPLATE(SC, longlong, long long)                                 \
    NVSHMEMI_FN_TEMPLATE(SC, uchar, unsigned char)                                \
    NVSHMEMI_FN_TEMPLATE(SC, ushort, unsigned short)                              \
    NVSHMEMI_FN_TEMPLATE(SC, uint, unsigned int)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, ulong, unsigned long)                                \
    NVSHMEMI_FN_TEMPLATE(SC, ulonglong, unsigned long long)                       \
    NVSHMEMI_FN_TEMPLATE(SC, int8, int8_t)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, int16, int16_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, int32, int32_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, int64, int64_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint8, uint8_t)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint16, uint16_t)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, uint32, uint32_t)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, uint64, uint64_t)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, size, size_t)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                         SC_PREFIX)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, float, float)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, double, double)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, char, char)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, short, short)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, schar, signed char)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int, int)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, long, long)                                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, longlong, long long)                       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uchar, unsigned char)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ushort, unsigned short)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint, unsigned int)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulong, unsigned long)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulonglong, unsigned long long)             \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int8, int8_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int16, int16_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int32, int32_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int64, int64_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint8, uint8_t)                            \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint16, uint16_t)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint32, uint32_t)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint64, uint64_t)                          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, size, size_t)                              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_AND_SCOPES2(NVSHMEMI_FN_TEMPLATE)             \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, thread, , )     \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, warp, _warp, x) \
    NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, block, _block, x)

#define NVSHMEMI_REPT_FOR_SIZES(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(8)                           \
    NVSHMEMI_FN_TEMPLATE(16)                          \
    NVSHMEMI_FN_TEMPLATE(32)                          \
    NVSHMEMI_FN_TEMPLATE(64)                          \
    NVSHMEMI_FN_TEMPLATE(128)

#define NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(8, int8_t)                             \
    NVSHMEMI_FN_TEMPLATE(16, int16_t)                           \
    NVSHMEMI_FN_TEMPLATE(32, int32_t)                           \
    NVSHMEMI_FN_TEMPLATE(64, int64_t)                           \
    NVSHMEMI_FN_TEMPLATE(128, int4)

#define NVSHMEMI_REPT_FOR_SIZES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SCOPE, SC_SUFFIX, SC_PREFIX) \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 8)                                       \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 6)                                       \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 32)                                      \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 64)                                      \
    NVSHMEMI_FN_TEMPLATE(SCOPE, SC_SUFFIX, SC_PREFIX, 128)

#define NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(short, short)                     \
    NVSHMEMI_FN_TEMPLATE(int, int)                         \
    NVSHMEMI_FN_TEMPLATE(long, long)                       \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)              \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)           \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)               \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)             \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)    \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                   \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                   \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                 \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                 \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                     \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMX_REPT_FOR_SIGNAL_TYPES(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(short, short)                       \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short)             \
    NVSHMEMI_FN_TEMPLATE(int, int)                           \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int)                 \
    NVSHMEMI_FN_TEMPLATE(long, long)                         \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long)               \
    NVSHMEMI_FN_TEMPLATE(longlong, long long)                \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long)      \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t)                     \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t)                     \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t)                   \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t)                   \
    NVSHMEMI_FN_TEMPLATE(size, size_t)                       \
    NVSHMEMI_FN_TEMPLATE(ptrdiff, ptrdiff_t)

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_FN_TEMPLATE(uchar, unsigned char, opname)                       \
    NVSHMEMI_FN_TEMPLATE(ushort, unsigned short, opname)                     \
    NVSHMEMI_FN_TEMPLATE(uint, unsigned int, opname)                         \
    NVSHMEMI_FN_TEMPLATE(ulong, unsigned long, opname)                       \
    NVSHMEMI_FN_TEMPLATE(ulonglong, unsigned long long, opname)              \
    NVSHMEMI_FN_TEMPLATE(int8, int8_t, opname)                               \
    NVSHMEMI_FN_TEMPLATE(int16, int16_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(int32, int32_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(int64, int64_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(uint8, uint8_t, opname)                             \
    NVSHMEMI_FN_TEMPLATE(uint16, uint16_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(uint32, uint32_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(uint64, uint64_t, opname)                           \
    NVSHMEMI_FN_TEMPLATE(size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)      \
    NVSHMEMI_FN_TEMPLATE(char, char, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(schar, signed char, opname)                          \
    NVSHMEMI_FN_TEMPLATE(short, short, opname)                                \
    NVSHMEMI_FN_TEMPLATE(int, int, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(long, long, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(longlong, long long, opname)                         \
    NVSHMEMI_FN_TEMPLATE(float, float, opname)                                \
    NVSHMEMI_FN_TEMPLATE(double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)  \
    NVSHMEMI_FN_TEMPLATE(complexf, double complex, opname)                 \
    NVSHMEMI_FN_TEMPLATE(complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, opname)
#endif

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_FN_TEMPLATE(SC, uchar, unsigned char, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, ushort, unsigned short, opname)                                \
    NVSHMEMI_FN_TEMPLATE(SC, uint, unsigned int, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, ulong, unsigned long, opname)                                  \
    NVSHMEMI_FN_TEMPLATE(SC, ulonglong, unsigned long long, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, int8, int8_t, opname)                                          \
    NVSHMEMI_FN_TEMPLATE(SC, int16, int16_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, int32, int32_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, int64, int64_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, uint8, uint8_t, opname)                                        \
    NVSHMEMI_FN_TEMPLATE(SC, uint16, uint16_t, opname)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint32, uint32_t, opname)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, uint64, uint64_t, opname)                                      \
    NVSHMEMI_FN_TEMPLATE(SC, size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)      \
    NVSHMEMI_FN_TEMPLATE(SC, char, char, opname)                                             \
    NVSHMEMI_FN_TEMPLATE(SC, schar, signed char, opname)                                     \
    NVSHMEMI_FN_TEMPLATE(SC, short, short, opname)                                           \
    NVSHMEMI_FN_TEMPLATE(SC, int, int, opname)                                               \
    NVSHMEMI_FN_TEMPLATE(SC, long, long, opname)                                             \
    NVSHMEMI_FN_TEMPLATE(SC, longlong, long long, opname)                                    \
    NVSHMEMI_FN_TEMPLATE(SC, float, float, opname)                                           \
    NVSHMEMI_FN_TEMPLATE(SC, double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)  \
    NVSHMEMI_FN_TEMPLATE(SC, complexf, double complex, opname)                            \
    NVSHMEMI_FN_TEMPLATE(SC, complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(NVSHMEMI_FN_TEMPLATE, SC, opname)
#endif

#define NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                           SC_PREFIX, opname)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uchar, unsigned char, opname)                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ushort, unsigned short, opname)              \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint, unsigned int, opname)                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulong, unsigned long, opname)                \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, ulonglong, unsigned long long, opname)       \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int8, int8_t, opname)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int16, int16_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int32, int32_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int64, int64_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint8, uint8_t, opname)                      \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint16, uint16_t, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint32, uint32_t, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, uint64, uint64_t, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, size, size_t, opname)

/* Note: The "long double" type is not supported */
#define NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                            SC_PREFIX, opname)                   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,      \
                                                       SC_PREFIX, opname)                        \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, char, char, opname)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, schar, signed char, opname)                   \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, short, short, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, int, int, opname)                             \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, long, long, opname)                           \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, longlong, long long, opname)                  \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, float, float, opname)                         \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, double, double, opname)

#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                         SC_PREFIX, opname)                   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,  \
                                                        SC_PREFIX, opname)                    \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexf, double complex, opname)          \
    NVSHMEMI_FN_TEMPLATE(SC, SC_SUFFIX, SC_PREFIX, complexd, float complex, opname)
#else
#define NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                         SC_PREFIX, opname)                   \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,  \
                                                        SC_PREFIX, opname)
#endif

#define NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(NVSHMEMI_FN_TEMPLATE)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, and)  \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, or)   \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, xor)  \
                                                                       \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, min) \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, max) \
                                                                       \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, prod)   \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_FN_TEMPLATE, sum)

#define NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX, \
                                                           SC_PREFIX)                           \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                       SC_PREFIX, and)                          \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                       SC_PREFIX, or)                           \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,     \
                                                       SC_PREFIX, xor)                          \
                                                                                                \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,    \
                                                        SC_PREFIX, min)                         \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,    \
                                                        SC_PREFIX, max)                         \
                                                                                                \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,       \
                                                     SC_PREFIX, prod)                           \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, SC, SC_SUFFIX,       \
                                                     SC_PREFIX, sum)

#define NVSHMEMI_REPT_TYPES_AND_OPS_AND_SCOPES2_FOR_REDUCE(NVSHMEMI_FN_TEMPLATE)             \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, thread, , )     \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, warp, _warp, x) \
    NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE_WITH_SCOPE2(NVSHMEMI_FN_TEMPLATE, block, _block, x)

#define NVSHMEMI_REPT_FOR_SCOPES2(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(thread, , )                    \
    NVSHMEMI_FN_TEMPLATE(warp, _warp, x)                \
    NVSHMEMI_FN_TEMPLATE(block, _block, x)

#define NVSHMEMI_REPT_FOR_SCOPE(NVSHMEMI_FN_TEMPLATE) \
    NVSHMEMI_FN_TEMPLATE(thread)                      \
    NVSHMEMI_FN_TEMPLATE(warp)                        \
    NVSHMEMI_FN_TEMPLATE(block)

#define NVSHMEMI_DECL_THREAD_IDX_warp() \
    ;                                   \
    int myIdx;                          \
    asm volatile("mov.u32  %0, %laneid;" : "=r"(myIdx));

#define NVSHMEMI_DECL_THREADGROUP_SIZE_warp()                           \
    ;                                                                   \
    int groupSize = ((blockDim.x * blockDim.y * blockDim.z) < warpSize) \
                        ? (blockDim.x * blockDim.y * blockDim.z)        \
                        : warpSize;

#define NVSHMEMI_DECL_THREAD_IDX_block() \
    ;                                    \
    int myIdx = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);

#define NVSHMEMI_DECL_THREADGROUP_SIZE_block() \
    ;                                          \
    int groupSize = (blockDim.x * blockDim.y * blockDim.z);

#define NVSHMEMI_DECL_THREAD_IDX_thread() \
    ;                                     \
    int myIdx = 0;

#define NVSHMEMI_DECL_THREADGROUP_SIZE_thread() \
    ;                                           \
    int groupSize = 1;

#define NVSHMEMI_SYNC_warp() \
    ;                        \
    __syncwarp();

#define NVSHMEMI_SYNC_block() \
    ;                         \
    __syncthreads();

#define NVSHMEMI_SYNC_thread() ;

enum nvshmemi_team_op_t { SYNC = 0, ALLTOALL, BCAST, FCOLLECT, REDUCE };
typedef enum nvshmemi_team_op_t nvshmemi_team_op_t;

enum { NVSHMEMI_HEAP_KIND_VIDMEM = 0, NVSHMEMI_HEAP_KIND_SYSMEM };

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
    NVSHMEMI_CALL_SITE_PROXY_ENFORCE_CONSISTENCY_AT_TARGET,
    NVSHMEMI_CALL_SITE_PROXY_GLOBAL_EXIT,
    NVSHMEMI_CALL_SITE_PROXY_QUIET,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GE,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_EQ,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_NE,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_GT,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_LT,
    NVSHMEMI_CALL_SITE_SIGNAL_WAIT_UNTIL_LE,
    NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_FLAG,
    NVSHMEMI_CALL_SITE_AMO_FETCH_WAIT_DATA,
    NVSHMEMI_CALL_SITE_G_WAIT_FLAG,
};

#define TIMEOUT_NCYCLES 1e10

#define NVSHMEM_TEAM_WORLD 0
#define NVSHMEM_TEAM_SHARED 1
#define NVSHMEMX_TEAM_NODE 2
#define NVSHMEMX_TEAM_SAME_MYPE_NODE 3
#define NVSHMEMI_TEAM_SAME_GPU 4
#define NVSHMEMI_TEAM_GPU_LEADERS 5
#define NVSHMEM_TEAM_INVALID -1

enum {
    NVSHMEM_TEAM_WORLD_INDEX = 0,
    NVSHMEM_TEAM_SHARED_INDEX,
    NVSHMEM_TEAM_NODE_INDEX,
    NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX,
    NVSHMEM_TEAM_SAME_GPU_INDEX,
    NVSHMEM_TEAM_GPU_LEADERS_INDEX,
    NVSHMEM_TEAMS_MIN
};

extern nvshmemi_device_state_t nvshmemi_device_state;
extern nvshmemi_pe_dist_t nvshmemi_pe_dist;
extern nvshmemi_team_t **nvshmemi_team_pool;

#if defined(__CUDACC_RDC__)
#define EXTERN_CONSTANT extern __constant__
#else
#define EXTERN_CONSTANT static __constant__
#endif

#undef EXTERN_CONSTANT

typedef enum {
    NVSHMEMI_JOB_GPU_LDST_ATOMICS = 1,
    NVSHMEMI_JOB_GPU_LDST = 1 << 1,
    NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS = 1 << 2,
    NVSHMEMI_JOB_GPU_PROXY = 1 << 3,
    NVSHMEMI_JOB_GPU_PROXY_CST = 1 << 4,
} nvshmemi_job_connectivity_t;

/* Start shared connectivity constants */
#define SYNC_SIZE 27648 /*XXX:Number of GPUs on Summit; currently O(N), need to be O(1)*/
#define NVSHMEMI_SYNC_SIZE (2 * SYNC_SIZE)
#define NVSHMEMI_BCAST_SYNC_SIZE (10 * SYNC_SIZE)
#define NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE SYNC_SIZE
#define NVSHMEMI_ALLTOALL_SYNC_SIZE SYNC_SIZE
#define NVSHMEMI_WARP_SIZE 32

typedef enum rdxn_ops {
    RDXN_OPS_AND = 0,
    RDXN_OPS_and = 0,
    RDXN_OPS_OR = 1,
    RDXN_OPS_or = 1,
    RDXN_OPS_XOR = 2,
    RDXN_OPS_xor = 2,
    RDXN_OPS_MIN = 3,
    RDXN_OPS_min = 3,
    RDXN_OPS_MAX = 4,
    RDXN_OPS_max = 4,
    RDXN_OPS_SUM = 5,
    RDXN_OPS_sum = 5,
    RDXN_OPS_PROD = 6,
    RDXN_OPS_prod = 6,
    RDXN_OPS_MAXLOC = 7,
    RDXN_OPS_maxloc
} rdxn_ops_t;

typedef enum {
    NVSHMEMI_PROXY_NONE = 0,
    NVSHMEMI_PROXY_MINIMAL = 1,
    NVSHMEMI_PROXY_FULL = 1 << 1,
} nvshmemi_proxy_status;

typedef struct {
    int major;
    int minor;
    int patch;
} nvshmemi_version_t;

NVSHMEMI_HOSTDEVICE_PREFIX int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe,
                                                          nvshmemi_team_t *dest_team);
NVSHMEMI_HOSTDEVICE_PREFIX long *nvshmemi_team_get_psync(nvshmemi_team_t *team,
                                                         nvshmemi_team_op_t op);
NVSHMEMI_HOSTDEVICE_PREFIX long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team);

/* Return -1 if `global_pe` is not in the given active set.
 * If `global_pe` is in the active set, return the PE index within this set. */
__host__ __device__ static inline int nvshmemi_pe_in_active_set(int global_pe, int PE_start,
                                                                int PE_stride, int PE_size) {
    int n = (global_pe - PE_start) / PE_stride;
    if (global_pe < PE_start || (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    else {
        return n;
    }
}

typedef void (*nvshmemx_device_lib_init_cb)(void **dev_state_ptr, void **transport_dev_state_ptr);

#ifdef __cplusplus
extern "C" {
#endif
int nvshmemx_host_init(int requested_thread_support, int *provided_thread_support,
                       unsigned int bootstrap_flags, nvshmemx_init_attr_t *bootstrap_attr,
                       nvshmemi_version_t nvshmem_device_lib_version,
                       nvshmemx_device_lib_init_cb cb);
void nvshmemx_host_finalize(void *device_ctx, void *transport_device_ctx);
#ifdef __cplusplus
}
#endif

#endif
