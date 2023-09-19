/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "internal/common/nvshmem_internal.h"
#include "device/pt-to-pt/proxy_device.cuh"
#ifdef NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#include "device/pt-to-pt/transfer_device.cuh"
#else
#include "device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif

#ifdef __CUDA_ARCH__
__device__ void nvshmemi_transfer_rma_nbi_translator(void *rptr, void *lptr,
                                                     rma_bytesdesc_t bytesdesc, int pe,
                                                     const nvshmemi_op_t desc) {
    switch (desc) {
        case NVSHMEMI_OP_PUT:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_PUT>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        case NVSHMEMI_OP_P:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_PUT>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        case NVSHMEMI_OP_GET:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_GET>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        case NVSHMEMI_OP_G:
            nvshmemi_transfer_rma_nbi<NVSHMEMI_THREADGROUP_THREAD, NVSHMEMI_OP_GET>(
                (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes), pe);
            break;
        default:
            printf("Incorrect argument to on-stream\n");
    }
}
#endif

__global__ void nvshmemi_proxy_rma_entrypoint(void *rptr, void *lptr, rma_bytesdesc_t bytesdesc,
                                              int pe, const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_rma_nbi_translator((void *)rptr, (void *)lptr, bytesdesc, pe, desc);
#endif
}

__global__ void nvshmemi_proxy_rma_entrypoint_blocking(void *rptr, void *lptr,
                                                       rma_bytesdesc_t bytesdesc, int pe,
                                                       const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_rma_nbi_translator((void *)rptr, (void *)lptr, bytesdesc, pe, desc);
    nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_THREAD>(true);
#endif
}

__global__ void nvshmemi_proxy_rma_signal_entrypoint(void *rptr, void *lptr,
                                                     rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_rma_nbi_translator((void *)rptr, (void *)lptr, bytesdesc, pe, desc);
    nvshmemi_transfer_amo_nonfetch((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
#endif
}

__global__ void nvshmemi_proxy_rma_signal_entrypoint_blocking(void *rptr, void *lptr,
                                                              rma_bytesdesc_t bytesdesc,
                                                              uint64_t *sig_addr, uint64_t signal,
                                                              int sig_op, int pe,
                                                              const nvshmemi_op_t desc) {
#ifdef __CUDA_ARCH__
    nvshmemi_transfer_put_signal<NVSHMEMI_THREADGROUP_THREAD>(
        (void *)rptr, (void *)lptr, (size_t)(bytesdesc.nelems * bytesdesc.elembytes),
        (void *)sig_addr, signal, (nvshmemi_amo_t)sig_op, pe, false);
    nvshmemi_transfer_quiet<NVSHMEMI_THREADGROUP_THREAD>(true);
#endif
}

int nvshmemi_proxy_rma_launcher(void *args[], cudaStream_t cstrm, bool is_nbi, bool is_signal) {
    if (is_signal && is_nbi) {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_signal_entrypoint, 1, 1, args, 0,
                                cstrm);
    } else if (is_nbi) {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_entrypoint, 1, 1, args, 0, cstrm);
    } else if (is_signal) {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_signal_entrypoint_blocking, 1, 1,
                                args, 0, cstrm);
    } else {
        return cudaLaunchKernel((const void *)nvshmemi_proxy_rma_entrypoint_blocking, 1, 1, args, 0,
                                cstrm);
    }
}
