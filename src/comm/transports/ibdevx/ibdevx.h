/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _IBRC_H
#define _IBRC_H

#include "transport.h"
#include <linux/types.h>

#define NVSHMEMT_IBDEVX_DBSIZE 8
/* 64 bytes per WQE BB shift = log2(64) for easy multiplication. */
#define NVSHMEMT_IBDEVX_WQE_BB_SHIFT 6

/* Atomic mode for our transport */
#define NVSHMEMT_IBDEVX_MLX5_QPC_ATOMIC_MODE_UP_TO_64B  0x3

#define NVSHMEMT_IBDEVX_MLX5_SEND_WQE_DS 0x10

/* Indicates to DEVX that we should be using an SRQ. */
#define NVSHMEMT_IBDEVX_SRQ_TYPE_VALUE 0x1

#ifndef MLX5DV_UAR_ALLOC_TYPE_BF
#define MLX5DV_UAR_ALLOC_TYPE_BF 0x1
#endif

/* Enables remote read/write/atomic access for a QP */
#define NVSHMEMT_IBDEVX_INIT2R2R_PARAM_MASK 0xE

/* Important byte masks. */
#define NVSHMEMT_IBDEVX_MASK_UPPER_BYTE_32 0x00FFFFFF
#define NVSHMEMT_IBDEVX_MASK_LOWER_3_BYTES_32 0xFF000000

/* OPMOD Constants for AMOs. */
#define NVSHMEMT_IBDEVX_4_BYTE_EXT_AMO_OPMOD 0x08000000
#define NVSHMEMT_IBDEVX_8_BYTE_EXT_AMO_OPMOD 0x09000000

/* Mellanox (IEEE) vendor and device information */
#define MELLANOX_VENDOR_ID 0x02c9
#define MELLANOX_MIN_DEVICE_ID 4113
struct ibdevx_function_table {
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
    int (*query_gid)(struct ibv_context *context, uint8_t port_num,
                  int index, union ibv_gid *gid);
};

#endif
