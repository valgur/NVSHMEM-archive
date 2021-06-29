/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_coll.h"
#include <cstdio>
#include <cassert>

#ifdef __CUDA_ARCH__

#define NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, \
                                               source, nelems, PE_root, PE_start, PE_stride,   \
                                               PE_size, pSync)                                 \
    do {                                                                                       \
        int i;                                                                                 \
        int stride = PE_stride;                                                                \
        int PE_end = PE_start + (stride * PE_size);                                            \
        if (PE_root == nvshmemi_device_state_d.mype) {                                         \
            for (i = PE_start; i < PE_end; i += stride)                                        \
                nvshmem##SC_PREFIX##_##TYPENAME##_put_nbi##SC_SUFFIX(dest, source, nelems, i); \
        }                                                                                      \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(PE_start, PE_stride, PE_size, pSync, NULL);   \
    } while (0)

#define NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_DIRECT(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE,  \
                                                      dest, source, nelems, PE_root, PE_start,   \
                                                      logPE_stride, PE_size, pSync)              \
    do {                                                                                         \
        int i;                                                                                   \
        int stride = PE_stride;                                                                  \
        int PE_end = PE_start + (stride * PE_size);                                              \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                         \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                   \
        TYPE *dst_ptr;                                                                           \
        if (PE_root == nvshmemi_device_state_d.mype) {                                           \
            for (i = PE_start; i < PE_end; i += stride) {                                        \
                dst_ptr = (TYPE *)nvshmem_ptr(dest, i);                                          \
                GPU_BITS_COPY_THREADGROUP_DIRECT(TYPENAME, TYPE, dst_ptr, source, nelems, myIdx, \
                                                 groupSize);                                     \
            }                                                                                    \
        }                                                                                        \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(PE_start, PE_stride, PE_size, pSync, NULL);     \
    } while (0)

#define NVSHMEMI_GPU_BCAST_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source,     \
                                       nelems, PE_root, PE_start, PE_stride, PE_size, pSync)       \
    do {                                                                                           \
        if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST)                     \
            NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL_DIRECT(SC, SC_SUFFIX, SC_PREFIX, TYPENAME,      \
                                                          TYPE, dest, source, nelems, PE_root,     \
                                                          PE_start, PE_stride, PE_size, pSync);    \
        else                                                                                       \
            NVSHMEMI_GPU_BCAST_THREADGROUP_PUT2ALL(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, \
                                                   source, nelems, PE_root, PE_start, PE_stride,   \
                                                   PE_size, pSync);                                \
    } while (0)

#define DEFN_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)         \
    __device__ void nvshmem##SC_PREFIX##i_##TYPENAME##_broadcast##SC_SUFFIX(                            \
        TYPE *dest, const TYPE *source, size_t nelems, int PE_root, int PE_start,                       \
        int PE_stride, int PE_size, long *pSync) {                                                      \
        NVSHMEMI_GPU_BCAST_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source, nelems,  \
                                       PE_root, PE_start, PE_stride, PE_size, pSync);                   \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP, thread, , )
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP, block, _block, x)
#undef DEFN_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP

#define DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE) \
    __device__ int nvshmem##SC_PREFIX##_##TYPENAME##_broadcast##SC_SUFFIX(                     \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int PE_root) {     \
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];                      \
        nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                                         \
        nvshmem##SC_PREFIX##i_##TYPENAME##_broadcast##SC_SUFFIX(                               \
            dest, source, nelems,                                                              \
            nvshmemi_team_translate_pe(teami, PE_root, &nvshmemi_device_state_d.team_world),   \
            teami->start, teami->stride, teami->size, nvshmemi_team_get_psync(teami, BCAST));  \
        return 0;                                                                              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP, thread, , )
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP, block, _block, x)
#undef DEFN_NVSHMEMX_TYPENAME_BROADCAST_THREADGROUP

#endif /* __CUDA_ARCH__ */
