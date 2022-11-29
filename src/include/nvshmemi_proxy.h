/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_PROXY_H_
#define _NVSHMEMI_PROXY_H_

template <typename T>
__device__ void nvshmemi_proxy_rma_p(void *rptr, T value, int pe);
__device__ void nvshmemi_proxy_rma_nbi(void *rptr, void *lptr, size_t nelems, int pe,
                                       nvshmemi_op_t channel_op);
template <typename T>
__device__ void nvshmemi_proxy_amo_nonfetch(void *rptr, const T value, int pe, nvshmemi_amo_t op);
template <typename T>
__device__ void nvshmemi_proxy_amo_fetch(void *rptr, void *lptr, T value, T compare, int pe,
                                         nvshmemi_amo_t op);
template <typename T>
__device__ T nvshmemi_proxy_rma_g(void *source, int pe);
__device__ void nvshmemi_proxy_fence();
__device__ void nvshmemi_proxy_quiet(bool use_membar);
__device__ void nvshmemi_proxy_global_exit(int status);
__device__ void nvshmemi_proxy_enforce_consistency_at_target(bool use_membar);

#endif /* _NVSHMEMI_PROXY_H_ */
