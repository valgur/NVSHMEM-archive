/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _TRANSPORT_COMMON_H
#define _TRANSPORT_COMMON_H

#include "nvshmem.h"
#include "nvshmem_internal.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "gdrapi.h"
#endif

#include <dlfcn.h>

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

struct nvshmemt_hca_info {
    char name[64];
    int port;
    int count;
    int found;
};

#ifdef NVSHMEM_USE_GDRCOPY
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

bool nvshmemt_gdrcopy_ftable_init(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle);
void nvshmemt_gdrcopy_ftable_fini(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle);
#endif

int nvshmemt_parse_hca_list(const char *string, struct nvshmemt_hca_info *hca_list, int max_count);
int nvshmemt_ib_iface_get_mlx_path(const char *ib_name, char **path);

struct nvshmemt_ibv_function_table {
    int (*fork_init)(void);
    struct ibv_ah *(*create_ah)(struct ibv_pd *pd, struct ibv_ah_attr *ah_attr);
    struct ibv_device **(*get_device_list)(int *num_devices);
    const char *(*get_device_name)(struct ibv_device *device);
    struct ibv_context *(*open_device)(struct ibv_device *device);
    int (*close_device)(struct ibv_context *context);
    int (*query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
    int (*query_port)(struct ibv_context *context, uint8_t port_num,
                      struct ibv_port_attr *port_attr);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *context);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
    struct ibv_mr *(*reg_dmabuf_mr)(struct ibv_pd *pd, uint64_t offset, size_t length,
                                    uint64_t iova, int fd, int access);
    int (*dereg_mr)(struct ibv_mr *mr);
    struct ibv_cq *(*create_cq)(struct ibv_context *context, int cqe, void *cq_context,
                                struct ibv_comp_channel *channel, int comp_vector);
    struct ibv_qp *(*create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
    struct ibv_srq *(*create_srq)(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr);
    int (*modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
    int (*query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
    int (*destroy_qp)(struct ibv_qp *qp);
    int (*destroy_cq)(struct ibv_cq *cq);
    int (*destroy_srq)(struct ibv_srq *srq);
    int (*destroy_ah)(struct ibv_ah *ah);
};

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable);
void nvshmemt_ibv_ftable_fini(void **ibv_handle);

#endif
