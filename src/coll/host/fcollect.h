/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_FCOLLECT_CPU_H
#define NVSHMEMI_FCOLLECT_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

template <typename TYPE>
void nvshmemi_call_fcollect_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                             size_t nelems, cudaStream_t stream);

#endif /* NVSHMEMI_FCOLLECT_CPU_H */
