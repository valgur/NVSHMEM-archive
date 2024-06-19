/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_IB_COMMON_H
#define _TRANSPORT_IB_COMMON_H

#include <stdint.h>                                              // for uint32_t
#include <string.h>                                              // for size_t
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvshmem_mem_handle_t
#include "transport_ib_common.h"                                 // lines 10-10

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

struct nvshmemt_ib_common_mem_handle {
    struct ibv_mr *mr;
    void *buf;
    int fd;
    uint32_t lkey;
    uint32_t rkey;
    bool local_only;
};

int nvshmemt_ib_common_nv_peer_mem_available();

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable, struct ibv_pd *pd,
                                      nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                      bool local_only, bool dmabuf_support,
                                      struct nvshmemi_cuda_fn_table *table, int log_level,
                                      bool relaxed_ordering);

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level);

#endif
