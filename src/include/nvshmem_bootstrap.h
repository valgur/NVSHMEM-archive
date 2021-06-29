/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef NVSHMEM_BOOTSTRAP_H
#define NVSHMEM_BOOTSTRAP_H

typedef struct bootstrap_handle {
    int pg_rank;
    int pg_size;
    int (*allgather)(const void *sendbuf, void *recvbuf, int bytes,
                     struct bootstrap_handle *handle);
    int (*alltoall)(const void *sendbuf, void *recvbuf, int bytes,
                     struct bootstrap_handle *handle);
    int (*barrier)(struct bootstrap_handle *handle);
    int (*finalize)(struct bootstrap_handle *handle);
} bootstrap_handle_t;

int nvshmemi_bootstrap_plugin_init(void *mpi_comm, bootstrap_handle_t *handle);

#endif
