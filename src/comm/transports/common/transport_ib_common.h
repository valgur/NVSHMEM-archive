/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_IB_COMMON_H
#define _TRANSPORT_IB_COMMON_H

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "transport_common.h"
#include "infiniband/verbs.h"

#ifdef NVSHMEM_IBDEVX_SUPPORT
#define NVSHMEM_MLX5_CODE
#endif

#ifdef NVSHMEM_IBGDA_SUPPORT
#define NVSHMEM_MLX5_CODE
#endif

#ifdef NVSHMEM_MLX5_CODE
#include "mlx5_ifc.h"
#include "mlx5_prm.h"
#include "infiniband/mlx5dv.h"
#endif

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

struct nvshmemt_ib_common_mem_handle {
    int fd;
    uint32_t lkey;
    uint32_t rkey;
    struct ibv_mr *mr;
};

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable, struct ibv_pd *pd,
                                      nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                      bool local_only, bool dmabuf_support);

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle);

#ifdef NVSHMEM_MLX5_CODE
bool nvshmemt_ib_common_query_mlx5_caps(struct ibv_context *context);
#endif

#endif