/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

template <nvshmemi_op_t desc, int is_nbi>
__global__ void nvshmemi_proxy_rma_entrypoint(void *rptr, void *lptr, rma_bytesdesc_t bytesdesc,
                                              int pe);

template <nvshmemi_op_t desc, int is_nbi>
__global__ void nvshmemi_proxy_rma_signal_entrypoint(void *rptr, void *lptr,
                                                     rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe);

template <>
__global__ void nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_PUT, 1>(void *rptr, void *lptr,
                                                      rma_bytesdesc_t bytesdesc, int pe) {
    nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)rptr, (void *)lptr,
                                       bytesdesc.nelems * bytesdesc.elembytes, pe);
}

template <>
__global__ void nvshmemi_proxy_rma_signal_entrypoint<NVSHMEMI_OP_PUT_SIGNAL, 1>(
    void *rptr, void *lptr, rma_bytesdesc_t bytesdesc, uint64_t *sig_addr, uint64_t signal,
    int sig_op, int pe) {
    nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)rptr, (void *)lptr,
                                            bytesdesc.nelems * bytesdesc.elembytes, pe);
    nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
}

template <>
__global__ void nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_GET, 1>(void *rptr, void *lptr,
                                                      rma_bytesdesc_t bytesdesc, int pe) {
    nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)rptr, (void *)lptr,
                                       bytesdesc.nelems * bytesdesc.elembytes, pe);
}

template <>
__global__ void nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_PUT, 0>(void *rptr, void *lptr,
                                                      rma_bytesdesc_t bytesdesc, int pe) {
    nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)rptr, (void *)lptr,
                                       bytesdesc.nelems * bytesdesc.elembytes, pe);
    nvshmemi_proxy_quiet(true);
}

template <>
__global__ void nvshmemi_proxy_rma_signal_entrypoint<NVSHMEMI_OP_PUT_SIGNAL, 0>(
    void *rptr, void *lptr, rma_bytesdesc_t bytesdesc, uint64_t *sig_addr, uint64_t signal,
    int sig_op, int pe) {
    nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)rptr, (void *)lptr,
                                            bytesdesc.nelems * bytesdesc.elembytes, pe);
    nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
    nvshmemi_proxy_quiet(true);
}

template <>
__global__ void nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_GET, 0>(void *rptr, void *lptr,
                                                      rma_bytesdesc_t bytesdesc, int pe) {
    nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)rptr, (void *)lptr,
                                       bytesdesc.nelems * bytesdesc.elembytes, pe);
    nvshmemi_proxy_quiet(true);
}

template <nvshmemi_op_t desc, int is_nbi>
int nvshmemi_proxy_rma_launcher(void *args[], cudaStream_t cstrm);

typedef void (*rma)(void *, void *, rma_bytesdesc_t, int);
typedef void (*rma_signal)(void *, void *, rma_bytesdesc_t, uint64_t *, uint64_t, int, int);

template <>
int nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_PUT, 1>(void *args[], cudaStream_t cstrm) {
    rma put_nbi = nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_PUT, 1>;
    return cudaLaunchKernel((const void *)put_nbi, 1, 1, args, 0, cstrm);
}

template <>
int nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_PUT_SIGNAL, 1>(void *args[], cudaStream_t cstrm) {
    rma_signal put_signal_nbi = nvshmemi_proxy_rma_signal_entrypoint<NVSHMEMI_OP_PUT_SIGNAL, 1>;
    return cudaLaunchKernel((const void *)put_signal_nbi, 1, 1, args, 0, cstrm);
}

template <>
int nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_GET, 1>(void *args[], cudaStream_t cstrm) {
    rma get_nbi = nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_GET, 1>;
    return cudaLaunchKernel((const void *)get_nbi, 1, 1, args, 0, cstrm);
}

template <>
int nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_PUT, 0>(void *args[], cudaStream_t cstrm) {
    rma put = nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_PUT, 0>;
    return cudaLaunchKernel((const void *)put, 1, 1, args, 0, cstrm);
}

template <>
int nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_PUT_SIGNAL, 0>(void *args[], cudaStream_t cstrm) {
    rma_signal put_signal = nvshmemi_proxy_rma_signal_entrypoint<NVSHMEMI_OP_PUT_SIGNAL, 0>;
    return cudaLaunchKernel((const void *)put_signal, 1, 1, args, 0, cstrm);
}

template <>
int nvshmemi_proxy_rma_launcher<NVSHMEMI_OP_GET, 0>(void *args[], cudaStream_t cstrm) {
    rma get = nvshmemi_proxy_rma_entrypoint<NVSHMEMI_OP_GET, 0>;
    return cudaLaunchKernel((const void *)get, 1, 1, args, 0, cstrm);
}
