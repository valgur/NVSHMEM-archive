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
    int mype_node;
    int npes_node;
    int (*allgather)(const void *sendbuf, void *recvbuf, int bytes,
                     struct bootstrap_handle *handle);
    int (*alltoall)(const void *sendbuf, void *recvbuf, int bytes, struct bootstrap_handle *handle);
    int (*barrier)(struct bootstrap_handle *handle);
    void (*global_exit)(int status);
    int (*finalize)(struct bootstrap_handle *handle);
} bootstrap_handle_t;

static bool nvshmemi_is_bootstrap_compatible(int bootstrap_version, int nvshmem_version) {
    if (bootstrap_version == nvshmem_version)
        return true;
    else
        return false;
}

#if __cplusplus
extern "C" {
#endif
int nvshmemi_bootstrap_plugin_init(void *mpi_comm, bootstrap_handle_t *handle,
                                   const int nvshmem_version);
#if __cplusplus
}
#endif

#endif
