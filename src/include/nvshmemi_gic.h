/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEM_GIC_H_
#define _NVSHMEM_GIC_H_

#include <linux/types.h>
#include <stdint.h>
#include <infiniband/mlx5dv.h>

#include "nvshmemi_util.h"

#define NVSHMEMI_GIC_CQE_SIZE 64
#define NVSHMEMI_GIC_MAX_INLINE_SIZE 8

typedef struct {
    int             lock;
    uint32_t        cqn;
    uint32_t        ncqes;
    void           *cqe;
    __be32         *dbrec;
    uint64_t       *cons_head;
    uint64_t       *cons_tail;
} nvshmemi_gic_device_cq_t;

// The ext flag (in dqp_dct) must be set to disable.
typedef struct {
    __be64		dc_key;
	__be32		dqp_dct;
	uint8_t		stat_rate_sl;
	uint8_t		fl_mlid;
	__be16		rlid;
} __attribute__((__packed__)) __attribute__((__aligned__(4)))
nvshmemi_gic_mlx5_wqe_half_av_t;

typedef struct {
    void           *wqe;
    __be32         *dbrec;
    void           *bf;
    nvshmemi_gic_device_cq_t *cq;
    uint16_t        nwqes;
    uint64_t        curr_idx;
    uint64_t        cons_head;
    uint64_t        cons_tail;
    uint64_t        get_head;
    uint64_t        get_tail;   // get_tail > get_head is possible
} nvshmemi_gic_device_wq_t;

#define NVSHMEMI_GIC_DS_MIN 2
#define NVSHMEMI_GIC_DS_MAX 6
#define nvshmemi_gic_ctrl_seg_ds_to_template_idx(ds) ((ds) - NVSHMEMI_GIC_DS_MIN)
#define NVSHMEMI_GIC_MAX_WQEBB_PER_WQE ((int)((NVSHMEMI_GIC_DS_MAX) + 4 - 1) / 4)
#define NVSHMEMI_GIC_MIN_NUM_BATCH_SIZE 2

typedef struct {
    int lock;
    uint32_t qpn;
    nvshmemi_gic_device_wq_t tx_wq;
    struct mlx5_wqe_ctrl_seg ctrl_seg_templates[NVSHMEMI_GIC_DS_MAX - NVSHMEMI_GIC_DS_MIN + 1];
    nvshmemi_gic_mlx5_wqe_half_av_t half_av_seg_template;
    struct {
        void *buf;  /* first uint64_t is for CST */
        __be32 lkey;
        __be32 rkey;
    } internal_buf;
} nvshmemi_gic_device_dci_t;

typedef struct {
    __be32      qpn;
    __be64      access_key;
    __be16      lid;
} nvshmemi_gic_device_dct_t;

typedef struct nvshmemi_gic_device_mhandle {
    union {
        __be32  lkey;   /* for local */
        __be32 *rkeys;  /* for remote; array of size npes */
    };
    uint64_t   start;
    uint64_t   end;
    struct nvshmemi_gic_device_mhandle *next;
} nvshmemi_gic_device_mhandle_t;

typedef struct {
    nvshmemi_gic_device_cq_t *cqs;
    nvshmemi_gic_device_dci_t *dcis;
    nvshmemi_gic_device_dct_t *dcts;
    nvshmemi_gic_device_mhandle_t *local_mhandle_head;
    nvshmemi_gic_device_mhandle_t *remote_mhandle_head;
    uint32_t ndcis;
    uint32_t ndcis_per_sm;
    uint32_t ndcts_per_pe;
    bool nic_buf_on_gpumem;
} nvshmemi_gic_device_state_t;

#ifdef __CUDA_ARCH__
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ void nvshmemi_gic_rma_nbi(void *rptr, void *lptr, size_t bytes, int pe);

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ void nvshmemi_gic_rma(void *rptr, void *lptr, size_t bytes, int pe);

template <typename T>
__device__ void nvshmemi_gic_rma_p(void *rptr, const T value, int pe);

template<typename T>
__device__ T nvshmemi_gic_rma_g(void *rptr, int pe);

template <typename T>
__device__ void nvshmemi_gic_amo_nonfetch(void *rptr, const T value, int pe, nvshmemi_amo_t op);

template <typename T>
__device__ T nvshmemi_gic_amo_fetch(void *rptr, const T value, const T compare, int pe, nvshmemi_amo_t op);

template <threadgroup_t SCOPE>
__device__ void nvshmemi_gic_put_signal(void *rptr, void *lptr, size_t bytes, 
    void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op, int pe, bool is_nbi);

__device__ void nvshmemi_gic_quiet();
__device__ void nvshmemi_gic_fence();
__device__ void nvshmemi_gic_enforce_consistency_at_target(bool use_membar);
#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEM_DEFINES_H_ */
