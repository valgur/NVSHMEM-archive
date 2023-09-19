/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "internal/common/comm/cuda_interface_sync.h"
#include "device/nvshmem_defines.h"
#include "device/nvshmemi_wait_until_apis.cuh"
#include "device/nvshmemi_common_device.cuh"
#include "internal/util.h"

#ifdef __CUDA_ARCH__
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL(TYPENAME, TYPE)                            \
    __global__ void nvshmemi_##TYPENAME##_wait_until_on_stream_kernel(volatile TYPE *ivar,       \
                                                                      int cmp, TYPE cmp_value) { \
        nvshmem_##TYPENAME##_wait_until((TYPE *)ivar, cmp, cmp_value);                           \
    }
#else
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL(TYPENAME, TYPE)                      \
    __global__ void nvshmemi_##TYPENAME##_wait_until_on_stream_kernel(volatile TYPE *ivar, \
                                                                      int cmp, TYPE cmp_value) {}
#endif
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL)
#undef NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL

#define CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL(TYPENAME, TYPE)                  \
    void call_nvshmemi_##TYPENAME##_wait_until_on_stream_kernel(                            \
        volatile TYPE *ivar, int cmp, TYPE cmp_value, cudaStream_t cstream) {               \
        nvshmemi_##TYPENAME##_wait_until_on_stream_kernel<<<1, 1, 0, cstream>>>(ivar, cmp,  \
                                                                                cmp_value); \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                             \
    }
NVSHMEMI_REPT_FOR_WAIT_TYPES(CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL)
#undef CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ON_STREAM_KERNEL

#ifdef __CUDA_ARCH__
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL(TYPENAME, TYPE)                   \
    __global__ void nvshmemi_##TYPENAME##_wait_until_all_on_stream_kernel(                  \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) {  \
        nvshmem_##TYPENAME##_wait_until_all((TYPE *)ivars, nelems, status, cmp, cmp_value); \
    }
#else
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL(TYPENAME, TYPE)  \
    __global__ void nvshmemi_##TYPENAME##_wait_until_all_on_stream_kernel( \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value) {}
#endif
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL)
#undef NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL

#define CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL(TYPENAME, TYPE)           \
    void call_nvshmemi_##TYPENAME##_wait_until_all_on_stream_kernel(                     \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, \
        cudaStream_t cstream) {                                                          \
        nvshmemi_##TYPENAME##_wait_until_all_on_stream_kernel<<<1, 1, 0, cstream>>>(     \
            ivars, nelems, status, cmp, cmp_value);                                      \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                          \
    }
NVSHMEMI_REPT_FOR_WAIT_TYPES(CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL)
#undef CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_ON_STREAM_KERNEL

#ifdef __CUDA_ARCH__
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL(TYPENAME, TYPE)                   \
    __global__ void nvshmemi_##TYPENAME##_wait_until_all_vector_on_stream_kernel(                  \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value) {        \
        nvshmem_##TYPENAME##_wait_until_all_vector((TYPE *)ivars, nelems, status, cmp, cmp_value); \
    }
#else
#define NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL(TYPENAME, TYPE)  \
    __global__ void nvshmemi_##TYPENAME##_wait_until_all_vector_on_stream_kernel( \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value) {}
#endif
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL)
#undef NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL

#define CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL(TYPENAME, TYPE)       \
    void call_nvshmemi_##TYPENAME##_wait_until_all_vector_on_stream_kernel(                 \
        volatile TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_value,   \
        cudaStream_t cstream) {                                                             \
        nvshmemi_##TYPENAME##_wait_until_all_vector_on_stream_kernel<<<1, 1, 0, cstream>>>( \
            ivars, nelems, status, cmp, cmp_value);                                         \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                             \
    }
NVSHMEMI_REPT_FOR_WAIT_TYPES(CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL)
#undef CALL_NVSHMEMI_TYPENAME_WAIT_UNTIL_ALL_VECTOR_ON_STREAM_KERNEL

__global__ void nvshmemi_signal_wait_until_on_stream_kernel(volatile uint64_t *sig_addr, int cmp,
                                                            uint64_t cmp_value) {
#ifdef __CUDA_ARCH__
    nvshmemi_wait_until<uint64_t>(sig_addr, cmp, cmp_value);
#endif
}

void call_nvshmemi_signal_wait_until_on_stream_kernel(volatile uint64_t *sig_addr, int cmp,
                                                      uint64_t cmp_value, cudaStream_t cstream) {
    nvshmemi_signal_wait_until_on_stream_kernel<<<1, 1, 0, cstream>>>(sig_addr, cmp, cmp_value);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

__global__ void nvshmemi_signal_op_kernel(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
#ifdef __CUDA_ARCH__
    nvshmemi_signal_op(sig_addr, signal, sig_op, pe);
#endif
}

void call_nvshmemi_signal_op_kernel(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                    cudaStream_t cstrm) {
    nvshmemi_signal_op_kernel<<<1, 1, 0, cstrm>>>(sig_addr, signal, sig_op, pe);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}
