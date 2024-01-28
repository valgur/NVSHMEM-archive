/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_IBGDA_COMMON_H_
#define _NVSHMEMI_IBGDA_COMMON_H_

#include <linux/types.h>
#include <stdint.h>
#include <infiniband/mlx5dv.h>

#define NVSHMEMI_IBGDA_MIN_QP_DEPTH 128
#define NVSHMEMI_IBGDA_MAX_QP_DEPTH 32768
#define NVSHMEMI_IBGDA_IBUF_SLOT_SIZE 256  // 32 threads * sizeof(uint64_t)

#define NVSHMEMI_IBGDA_MAX_CONST_LKEYS 64
#define NVSHMEMI_IBGDA_MAX_CONST_RKEYS 64
#define NVSHMEMI_IBGDA_MAX_CONST_DCTS 128

/* This is determined by the size of nvshmem_mem_handle_t*/
#define NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE 15

typedef enum {
    NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI = 1,
    NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT = 2,
    NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC = 3
} nvshmemi_ibgda_device_qp_type_t;

typedef enum {
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA = 0,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_DCT,
    NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID
} nvshmemi_ibgda_device_qp_map_type_t;

typedef struct {
    void *cqe;
    uint64_t *prod_idx;
    uint64_t *cons_idx;
    uint64_t *resv_head;
    uint64_t *ready_head;
    uint32_t cqn;
    uint32_t ncqes;
    uint32_t qpn;
    nvshmemi_ibgda_device_qp_type_t qp_type;
    __be32 *dbrec;
} nvshmemi_ibgda_device_cq_t;

// Variables for queue management.
// They are always in global memory.
typedef struct {
    int post_send_lock;
    struct {
        // All indexes are in wqebb unit
        uint64_t resv_head;   // last reserved wqe idx + 1
        uint64_t ready_head;  // last ready wqe idx + 1
        uint64_t prod_idx;    // posted wqe idx + 1 (producer index + 1)
        uint64_t cons_idx;    // polled wqe idx + 1 (consumer index + 1)
        uint64_t get_head;    // last wqe idx + 1 with a "fetch" operation (g, get, amo_fetch)
        uint64_t get_tail;    // last wqe idx + 1 polled with cst; get_tail > get_head is possible
    } tx_wq;
    struct {
        uint64_t head;
        uint64_t tail;
    } ibuf;
} __attribute__((__aligned__(8))) nvshmemi_ibgda_device_qp_management_t;

typedef struct nvshmemi_ibgda_device_qp {
    nvshmemi_ibgda_device_qp_type_t qp_type;
    uint32_t qpn;
    uint32_t dev_idx;
    struct {
        uint32_t nslots;  // num slots for fetch; always a power of 2
        void *buf;        // first NVSHMEMI_IBGDA_IBUF_SLOT_SIZE is for non-fetch
        __be32 lkey;
        __be32 rkey;
    } ibuf;  // Internal buffer
    struct {
        uint16_t nwqes;  // num wqes; some wqes may consume n wqebbs
        void *wqe;
        __be32 *dbrec;
        void *bf;
        nvshmemi_ibgda_device_cq_t *cq;
    } tx_wq;
    nvshmemi_ibgda_device_qp_management_t mvars;  // management variables
} nvshmemi_ibgda_device_qp_t;

typedef struct mlx5_wqe_av nvshmemi_ibgda_device_dct_t;

typedef struct nvshmemi_ibgda_device_local_only_mhandle {
    bool is_sysmem_scope;
    uint64_t start;
    uint64_t end;
    struct nvshmemi_ibgda_device_local_only_mhandle *next;
    __be32 lkeys[NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE];
} nvshmemi_ibgda_device_local_only_mhandle_t;

typedef struct {
    __be32 key;
    uint64_t next_addr;  // end of this address range + 1
} nvshmemi_ibgda_device_key_t;

typedef struct {
    size_t log2_cumem_granularity;
    uint32_t num_shared_dcis;
    uint32_t num_exclusive_dcis;
    nvshmemi_ibgda_device_qp_map_type_t dci_map_type;
    uint32_t ndcts_per_pe;
    uint32_t num_qp_groups;
    uint32_t num_dct_groups;
    uint32_t num_rc_per_pe;
    nvshmemi_ibgda_device_qp_map_type_t rc_map_type;
    uint32_t num_requests_in_batch; /* always a power of 2 */
    int num_devices_initialized;
    bool nic_buf_on_gpumem;
    bool support_half_av_seg;
    bool may_skip_cst;

    struct {
        uint8_t *qp_group_switches;
        nvshmemi_ibgda_device_cq_t *cqs;  // For both dcis and rcs. CQs for DCIs come first.
        nvshmemi_ibgda_device_qp_t *dcis;
        nvshmemi_ibgda_device_qp_t *rcs;
        nvshmemi_ibgda_device_local_only_mhandle *local_only_mhandle_head;

        // For dcts that cannot be contained in constmem.lkeys.
        // dcts[idx - NVSHMEMI_IBGDA_MAX_CONST_DCTS] gives the dct of idx.
        nvshmemi_ibgda_device_dct_t *dcts;

        // For lkeys that cannot be contained in constmem.lkeys.
        // lkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_LKEYS] gives the lkey of chunk idx.
        nvshmemi_ibgda_device_key_t *lkeys;

        // For rkeys that cannot be contained in constmem.rkeys.
        // rkeys[(idx * npes + pe) - NVSHMEMI_IBGDA_MAX_CONST_RKEYS] gives rkey of chunck idx
        // targeting peer pe.
        nvshmemi_ibgda_device_key_t *rkeys;
    } globalmem;

    struct {
        // lkeys[idx] gives the lkey of chunk idx.
        nvshmemi_ibgda_device_key_t lkeys[NVSHMEMI_IBGDA_MAX_CONST_LKEYS];

        // rkeys[idx * npes + pe] gives rkey of chunck idx targeting peer pe.
        nvshmemi_ibgda_device_key_t rkeys[NVSHMEMI_IBGDA_MAX_CONST_RKEYS];

        nvshmemi_ibgda_device_dct_t dcts[NVSHMEMI_IBGDA_MAX_CONST_DCTS];
    } constmem;
} nvshmemi_ibgda_device_state_t;

#if defined(__CUDACC_RDC__)
#define EXTERN_CONSTANT extern __constant__
EXTERN_CONSTANT nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;
#undef EXTERN_CONSTANT
#endif

#endif /* _NVSHMEMI_IBGDA_COMMON_H_ */
