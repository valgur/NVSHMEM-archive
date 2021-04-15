/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _UCX_H
#define _UCX_H

#include "transport.h"
#include "nvshmem.h"
#include "nvshmem_internal.h"

#include "gdrapi.h"

#include <ucs/type/status.h>
#include <ucp/api/ucp_def.h>
#include <ucp/api/ucp.h>
#include <deque>
#include <dlfcn.h>

/* This value is arbitrary. UCX doesn't give a max length for packed rkeys. */
#define NVSHMEMT_UCP_RKEY_PACKED_MAX_LEN 256

#define NVSHMEMT_UCP_ADDR_MAX_LEN 1024

#define NVSHMEM_UCX_ATOMIC_POOL_SIZE (1 << 14)

#define NVSHMEM_UCX_ATOMIC_POOL_MASK (NVSHMEM_UCX_ATOMIC_POOL_SIZE - 1)

#define NVSHMEM_UCX_P_BUFFER_POOL_SIZE (1 << 16)

#define NVSHMEM_UCX_P_BUFFER_POOL_MASK (NVSHMEM_UCX_P_BUFFER_POOL_SIZE - 1)

typedef enum {
    NVSHMEM_UCX_ATOMIC_SEND,
    NVSHMEM_UCX_ATOMIC_RESP,
} nvshmem_ucx_am_op_t;

typedef struct {
    ucp_ep_h        ep;
    size_t          op_size;
    uint64_t        value;
    uint64_t        cmp;
    uint64_t        retflag;
    void            *addr;
    void            *retptr;
    nvshmemi_amo_t  op;
} nvshmemt_ucx_am_send_header_t;

typedef struct {
    void        *retptr;
    uint64_t    retval;
    uint64_t    retflag;
} nvshmemt_ucx_am_resp_header_t;

typedef struct {
    union {
        nvshmemt_ucx_am_resp_header_t   resp_h;
        nvshmemt_ucx_am_send_header_t   send_h;
    }                                   header;
    bool                                is_proxy;
    bool                                in_use;
    bool                                nvshmem_owned;
    bool                                dynamic_alloc;
} nvshmemt_ucx_am_header_t;

typedef struct {
    char    addr[NVSHMEMT_UCP_ADDR_MAX_LEN];
    int     addr_len;
} ucx_ep_handle_t;

typedef struct {
    char        rkey_packed_buf[NVSHMEMT_UCP_RKEY_PACKED_MAX_LEN];
    ucp_mem_h   mem_handle;
    size_t      rkey_packed_buf_len;
} nvshmemt_ucx_mem_handle_t;

typedef struct {
    gdr_mh_t    mh;
    void        *cpu_ptr;
    void        *cpu_ptr_base;
    void        *ptr;
    size_t      size;
} nvshmemt_ucx_mem_handle_info_t;

typedef struct {
    union {
        uint8_t     buffer_1_byte;
        uint16_t    buffer_2_byte;
        uint32_t    buffer_4_byte;
        uint64_t    buffer_8_byte;
    }               buffer;
    bool            in_use;
} nvshmemt_ucx_p_buffer_t;

typedef struct {
    ucp_config_t                    *library_config;
    nvshmemt_ucx_mem_handle_info_t  mem_handle_info;
    ucp_context_h                   library_context;
    ucp_worker_h                    worker_context;
    ucp_ep_h                        *endpoints;
    ucp_rkey_h                      *ep_rkeys;
    int                             ep_count;
    int                             proxy_ep_idx;
    int                             num_headers_requested;
    uint16_t                        num_p_buffers_requested;
    nvshmemt_ucx_am_header_t        send_headers[NVSHMEM_UCX_ATOMIC_POOL_SIZE];
    nvshmemt_ucx_am_header_t        recv_headers[NVSHMEM_UCX_ATOMIC_POOL_SIZE];
    nvshmemt_ucx_p_buffer_t         p_buffers[NVSHMEM_UCX_P_BUFFER_POOL_SIZE];
} transport_ucx_state_t;

struct gdrcopy_function_table {
    gdr_t (*open)();
    int (*close)(gdr_t g);
    int (*pin_buffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token,
                      uint32_t va_space, gdr_mh_t *handle);
    int (*unpin_buffer)(gdr_t g, gdr_mh_t handle);
    int (*get_info)(gdr_t g, gdr_mh_t handle, gdr_info_t *info);
    int (*map)(gdr_t g, gdr_mh_t handle, void **va, size_t size);
    int (*unmap)(gdr_t g, gdr_mh_t handle, void *va, size_t size);
    int (*copy_from_mapping)(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);
    int (*copy_to_mapping)(gdr_mh_t handle, const void *map_d_ptr, void *h_ptr, size_t size);
    void (*runtime_get_version)(int *major, int *minor);
    int (*driver_get_version)(gdr_t g, int *major, int *minor);
};

#endif /* _UCX_H */