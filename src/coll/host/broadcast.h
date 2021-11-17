/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_BCAST_COMMON_CPU_H
#define NVSHMEMI_BCAST_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
void nvshmemi_call_broadcast_on_stream_kernel(nvshmem_team_t team, T *dest, const T *source,
                                              size_t nelems, int PE_root, cudaStream_t stream);
#endif
