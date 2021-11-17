/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_RDXN_COMMON_CPU_H
#define NVSHMEMI_RDXN_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_rdxn_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                         size_t nreduce, cudaStream_t stream);

#endif /* NVSHMEMI_RDXN_COMMON_CPU_H */
