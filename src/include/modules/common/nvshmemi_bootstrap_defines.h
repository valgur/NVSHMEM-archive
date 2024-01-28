/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef NVSHMEMI_BOOTSTRAP_DEFINES_H
#define NVSHMEMI_BOOTSTRAP_DEFINES_H

typedef struct bootstrap_env_attr {
    char *uid_session_id;
    char *uid_socket_ifname;
    char *uid_socket_family;
} bootstrap_env_attr_t;

typedef struct bootstrap_init_ops {
    void *cookie;
    bootstrap_env_attr_t *env_attr;
    int (*get_unique_id)(void *cookie, struct bootstrap_env_attr *attr);
} bootstrap_init_ops_t;

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
    bootstrap_init_ops_t *pre_init_ops;
    void *comm_state;
} bootstrap_handle_t;

#endif
