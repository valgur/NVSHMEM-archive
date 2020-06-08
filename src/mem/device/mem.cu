/*
 * * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#ifdef __CUDA_ARCH__

__device__ void *nvshmem_ptr(void *ptr, int pe) {
    ptrdiff_t offset = (char*)ptr - (char*)nvshmemi_heap_base_d;

    if (ptr >= nvshmemi_heap_base_d && offset < nvshmemi_heap_size_d) {
        void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
        return (void *)((char *)peer_base_addr + offset);
    }
    else
        return NULL;
}

#endif
