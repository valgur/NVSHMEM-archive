/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "infiniband/mlx5dv.h"

#include "nvshmem_internal.h"
#include "nvshmemx_error.h"
#include "nvshmemi_util.h"
#include "nvshmemi_gic.h"
#include "utils.h"

#define GIC_FULL_WARP 0xffffffffU
#define GIC_POLL_TIMEOUT 4000000000LLU

#ifndef likely
    #define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
    #define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef ACCESS_ONCE
    #define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

#ifndef READ_ONCE
    #define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
    #define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

#ifdef NVSHMEM_GPUINITIATED_DEBUG
struct mlx5_err_cqe_ex {
    uint8_t     rsvd0[32];
    uint32_t    srqn;
    uint8_t     rsvd1[16];
    uint8_t     hw_err_synd;
    uint8_t     hw_synd_type;
    uint8_t     vendor_err_synd;
    uint8_t     syndrome;
    uint32_t    s_wqe_opcode_qpn;
    uint16_t    wqe_counter;
    uint8_t     signature;
    uint8_t     op_own;
};
typedef struct mlx5_err_cqe_ex gic_mlx5_err_cqe_t;
#else
typedef struct mlx5_err_cqe gic_mlx5_err_cqe_t;
#endif

#define GIC_4_BYTE_EXT_AMO_OPMOD 0x08000000
#define GIC_8_BYTE_EXT_AMO_OPMOD 0x09000000

enum {
    GIC_MLX5_OPCODE_DUMP = 0x23,
};

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__))
gic_atomic_32_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_32_masked_fa_seg_t) == 16, "sizeof(gic_atomic_32_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t add_data;
    uint64_t field_boundary;
} __attribute__((__packed__))
gic_atomic_64_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_64_masked_fa_seg_t) == 16, "sizeof(gic_atomic_64_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint32_t swap_data;
    uint32_t compare_data;
    uint32_t swap_mask;
    uint32_t compare_mask;
} __attribute__((__packed__)) 
gic_atomic_32_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_32_masked_cs_seg_t) == 16, "sizeof(gic_atomic_32_masked_cs_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t swap;
    uint64_t compare;
} __attribute__((__packed__)) 
gic_atomic_64_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_atomic_64_masked_cs_seg_t) == 16, "sizeof(gic_atomic_64_masked_cs_seg_t) == 16 failed.");
#endif

typedef struct {
    struct mlx5_wqe_ctrl_seg ctrl_seg;
    nvshmemi_gic_mlx5_wqe_half_av_t av_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;
} __attribute__((__packed__)) __attribute__((__aligned__(4)))
gic_wqe_dc_rdma_rw_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_wqe_dc_rdma_rw_t) == 64, "sizeof(gic_wqe_dc_rdma_rw_t) == 64 failed.");
#endif

typedef struct {
    struct mlx5_wqe_ctrl_seg ctrl_seg;
    nvshmemi_gic_mlx5_wqe_half_av_t av_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;
    // Inline data is included after inl_seg
} __attribute__((__packed__)) __attribute__((__aligned__(4)))
gic_wqe_dc_rdma_write_inl_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_wqe_dc_rdma_write_inl_t) == 52, "sizeof(gic_wqe_dc_rdma_write_inl_t) == 52 failed.");
#endif

typedef struct {
    struct mlx5_wqe_ctrl_seg ctrl_seg;
    nvshmemi_gic_mlx5_wqe_half_av_t av_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    union {
        struct mlx5_wqe_atomic_seg atomic_seg;
        gic_atomic_32_masked_fa_seg_t atomic_32_masked_fa_seg;
        gic_atomic_32_masked_cs_seg_t atomic_32_masked_cs_seg;
        gic_atomic_64_masked_fa_seg_t atomic_64_masked_fa_seg;
    };
    struct mlx5_wqe_data_seg data_seg;
} __attribute__((__packed__)) __attribute__((__aligned__(4)))
gic_wqe_dc_atomic_5ds_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_wqe_dc_atomic_5ds_t) == 80, "sizeof(gic_wqe_dc_atomic_5ds_t) == 80 failed.");
#endif

typedef struct {
    struct mlx5_wqe_ctrl_seg ctrl_seg;
    nvshmemi_gic_mlx5_wqe_half_av_t av_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    gic_atomic_64_masked_cs_seg_t atomic_64_masked_cs_data_seg;
    gic_atomic_64_masked_cs_seg_t atomic_64_masked_cs_mask_seg;
    struct mlx5_wqe_data_seg data_seg;
} __attribute__((__packed__)) __attribute__((__aligned__(4)))
gic_wqe_dc_atomic_6ds_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_wqe_dc_atomic_6ds_t) == 96, "sizeof(gic_wqe_dc_atomic_6ds_t) == 96 failed.");
#endif

typedef struct {
    struct mlx5_wqe_ctrl_seg ctrl_seg;
    struct mlx5_wqe_data_seg data_seg;
} __attribute__((__packed__)) __attribute__((__aligned__(4)))
gic_wqe_dc_dump_t;
#if __cplusplus >= 201103L
static_assert(sizeof(gic_wqe_dc_dump_t) == 32, "sizeof(gic_wqe_dc_dump_t) == 32 failed.");
#endif

typedef union {
    gic_wqe_dc_rdma_rw_t dc_rdma_write;
    gic_wqe_dc_rdma_rw_t dc_rdma_read;
    gic_wqe_dc_rdma_write_inl_t dc_rdma_write_inl;
    gic_wqe_dc_atomic_5ds_t dc_atomic_5ds;
    gic_wqe_dc_atomic_6ds_t dc_atomic_6ds;
    gic_wqe_dc_dump_t dc_dump;
} gic_wqe_t;


#define GIC_REPT_FOR_STANDARD_RMA_TYPES(FN_TEMPLATE) \
    FN_TEMPLATE(char) \
    FN_TEMPLATE(unsigned char) \
    FN_TEMPLATE(short) \
    FN_TEMPLATE(unsigned short) \
    FN_TEMPLATE(int) \
    FN_TEMPLATE(unsigned int) \
    FN_TEMPLATE(long) \
    FN_TEMPLATE(unsigned long) \
    FN_TEMPLATE(long long) \
    FN_TEMPLATE(unsigned long long) \
    FN_TEMPLATE(float) \
    FN_TEMPLATE(double) 

#define GIC_REPT_FOR_STANDARD_AMO_TYPES(FN_TEMPLATE) \
    FN_TEMPLATE(short) \
    FN_TEMPLATE(unsigned short) \
    FN_TEMPLATE(int) \
    FN_TEMPLATE(unsigned int) \
    FN_TEMPLATE(long) \
    FN_TEMPLATE(unsigned long) \
    FN_TEMPLATE(long long) \
    FN_TEMPLATE(unsigned long long)

#define GIC_REPT_FOR_EXTENDED_AMO_TYPES(FN_TEMPLATE) \
    FN_TEMPLATE(float) \
    FN_TEMPLATE(double)

#define GIC_REPT_FOR_ALL_SCOPES(FN_TEMPLATE) \
    FN_TEMPLATE(NVSHMEMI_THREADGROUP_THREAD) \
    FN_TEMPLATE(NVSHMEMI_THREADGROUP_WARP) \
    FN_TEMPLATE(NVSHMEMI_THREADGROUP_BLOCK)


#ifdef __CUDA_ARCH__

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
__device__ static inline uint64_t gic_query_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}
#endif

__device__ static inline nvshmemi_gic_device_state_t *gic_get_state() {
    return (nvshmemi_gic_device_state_t *)nvshmemi_device_state_d.gic_state;
}

__device__ static inline void GIC_MFENCE() {
    if (gic_get_state()->nic_buf_on_gpumem)
        __threadfence();
    else
        __threadfence_system();
}

__device__ static inline uint32_t gic_get_smid() {
    uint32_t smid;
    asm volatile("mov.u32  %0, %smid;" : "=r"(smid));
    return smid;
}

__device__ static inline bool gic_try_lock(int *lock) {
    bool ret = (atomicCAS(lock, 0, 1) == 0);

    if (ret)
        __threadfence_block();  // Prevent reordering before lock is acquired.

    return ret;
}

template <threadgroup_t SCOPE>
__device__ static inline void gic_lock(int *lock) {
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0)
        while (atomicCAS(lock, 0, 1) == 1) ;    // Wait until we get the lock.

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        __threadfence_block();  // Prevent reordering before lock is acquired.

    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ static inline void gic_unlock(int *lock) {
    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        __threadfence_block();  // Prevent reordering before lock is released.

    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0)
        // atomicSet/Exch is not required because of the address of lock is properly aligned.
        WRITE_ONCE(*lock, 0);
}

__device__ static int gic_poll_cq(nvshmemi_gic_device_cq_t *cq, uint64_t cqe_idx, bool block, int *error) {
    uint32_t ncqes = cq->ncqes;
    uint32_t idx = cqe_idx & (ncqes - 1);

    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)((uintptr_t)cq->cqe + idx * NVSHMEMI_GIC_CQE_SIZE);

    #ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    uint64_t start = gic_query_globaltimer();
    uint64_t now;
    #endif

    do {
        uint8_t opown = READ_ONCE(cqe64->op_own); 
        uint8_t opcode = opown >> 4;

        if ((opcode != MLX5_CQE_INVALID) && !((opown & MLX5_CQE_OWNER_MASK) ^ !!(cqe_idx & ncqes))) {
            if (unlikely(opcode == MLX5_CQE_REQ_ERR)) {
                gic_mlx5_err_cqe_t *cqe_err = (gic_mlx5_err_cqe_t *)cqe64;
                *error = cqe_err->syndrome;
                #ifdef NVSHMEM_GPUINITIATED_DEBUG
                printf(
                    "got completion with err:"
                    "   syndrome=%#x, vendor_err_synd=%#x, hw_err_synd=%#x, hw_synd_type=%#x,"
                    "   wqe_counter=%u, s_wqe_opcode_qpn=%#x\n",
                    cqe_err->syndrome,
                    cqe_err->vendor_err_synd,
                    cqe_err->hw_err_synd,
                    cqe_err->hw_synd_type,
                    cqe_err->wqe_counter,
                    cqe_err->s_wqe_opcode_qpn
                );
                #endif
                return -1;
            }
            return 1;
        }
        #ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        // TODO: Integrate timeout handler with the core NVSHMEM
        now = gic_query_globaltimer();
        if (now - start > GIC_POLL_TIMEOUT) {
            *error = -ETIME;
            #ifdef NVSHMEM_GPUINITIATED_DEBUG
            printf("gic_poll_cq timeout.\n");
            #endif
            return -1;
        }
        #endif
    } while (block);

    return 0;
}

__device__ static inline void gic_cq_update_dbr(nvshmemi_gic_device_cq_t *cq, uint64_t cqe_idx) {
    // Update to the consumer index.
    __be32 *dbrec_ptr = cq->dbrec;
    __be32 dbrec_val = HTOBE32((uint32_t)cqe_idx & 0xffffff);

    WRITE_ONCE(*dbrec_ptr, dbrec_val);

    // tail is the index of the next not-yet-polled CQE
    WRITE_ONCE(*cq->cons_tail, cqe_idx);
}

__device__ static uint64_t gic_quiet(nvshmemi_gic_device_dci_t *dci) {
    uint64_t ticket = READ_ONCE(dci->tx_wq.cons_head);
    uint64_t tail;
    nvshmemi_gic_device_cq_t *cq = dci->tx_wq.cq;

    do {
        if (gic_try_lock(&cq->lock)) {
            int err = 0;
            int status = 0;
            tail = READ_ONCE(dci->tx_wq.cons_tail);
            if (tail < ticket) {
                status = gic_poll_cq(cq, ticket - 1, true, &err);
                // TODO: Integrate the error handler with the core NVSHMEM
                #ifdef NVSHMEM_GPUINITIATED_DEBUG
                if (status != 1) {
                    printf("gic_poll_cq failed.\n");
                }
                #endif
                assert(status == 1);
                gic_cq_update_dbr(cq, ticket);
                GIC_MFENCE();
            }
            gic_unlock<NVSHMEMI_THREADGROUP_THREAD>(&cq->lock);
            return ticket;
        }
        tail = READ_ONCE(dci->tx_wq.cons_tail);
    } while (tail < ticket);

    return ticket;
}

__device__ static inline void gic_ensure_wqe_availability(nvshmemi_gic_device_dci_t *dci, uint16_t num_requests) {
    // We first read head and tail from cache
    uint64_t head = dci->tx_wq.cons_head;
    uint64_t tail = dci->tx_wq.cons_tail;
    uint16_t nwqes = dci->tx_wq.nwqes;

    if (likely((head - tail + num_requests) * NVSHMEMI_GIC_MAX_WQEBB_PER_WQE <= nwqes))
        return;

    // TODO: Make it more efficient so that we are not exposed to full latency.
    gic_quiet(dci);
}


/**
 * Get a DCI.
 * Lock it if necessary.
 */
template <threadgroup_t SCOPE>
__device__ static inline nvshmemi_gic_device_dci_t *gic_get_dci(nvshmemi_gic_device_state_t *state) {
    nvshmemi_gic_device_dci_t *dci = NULL;

    uint32_t smid = gic_get_smid();
    uint32_t warpid = nvshmemi_thread_id_in_block() / warpSize;

    uint32_t idx = smid * state->ndcis_per_sm + MIN(warpid, state->ndcis_per_sm - 1);

    if (likely(idx < state->ndcis - 1)) {
        // Get sm-exclusive DCI
        dci = &state->dcis[idx];
    } else {
        // Get the shared DCI.
        dci = &state->dcis[state->ndcis - 1];
    }
    gic_lock<SCOPE>(&dci->lock);
    return dci;
}

template <threadgroup_t SCOPE>
__device__ static inline void gic_release_dci(nvshmemi_gic_device_dci_t *dci) {
    gic_unlock<SCOPE>(&dci->lock);
}

__device__ static inline nvshmemi_gic_device_dct_t *gic_get_dct(nvshmemi_gic_device_state_t *state, int pe) {
    uint32_t smid = gic_get_smid();
    int dct_idx = (pe * state->ndcts_per_pe) + (smid % state->ndcts_per_pe);

    return &state->dcts[dct_idx];
}

__device__ static inline __be32 gic_get_lkey(nvshmemi_gic_device_state_t *state, uint64_t addr) {
    nvshmemi_gic_device_mhandle_t *mhandle = state->local_mhandle_head;
    while (mhandle) {
        if (mhandle->start <= addr && addr <= mhandle->end)
            return mhandle->lkey;
        mhandle = mhandle->next;
    }

    // lkey is not found. 
    assert(0);
    return 0;
}

__device__ static inline void gic_get_raddr_rkey(nvshmemi_gic_device_state_t *state, uint64_t addr, int pe, uint64_t *raddr, __be32 *rkey) {
    nvshmemi_gic_device_mhandle_t *mhandle = state->remote_mhandle_head;
    uint64_t roffset = addr - (uint64_t)nvshmemi_device_state_d.heap_base;
    while (mhandle) {
        if (mhandle->start <= roffset && roffset <= mhandle->end) {
            *raddr = (uint64_t)nvshmemi_device_state_d.peer_heap_base_actual[pe] + roffset;
            *rkey = mhandle->rkeys[pe];
            return;
        }
        mhandle = mhandle->next;
    }

    // rkey is not found. 
    assert(0);
}

__device__ static inline uint64_t gic_get_current_wqe_idx(nvshmemi_gic_device_dci_t *dci) {
    return READ_ONCE(dci->tx_wq.curr_idx);
}

__device__ static inline void gic_update_current_wqe_idx(nvshmemi_gic_device_dci_t *dci, uint64_t new_idx) {
    WRITE_ONCE(dci->tx_wq.curr_idx, new_idx);
}

__device__ static inline uint64_t gic_get_current_cons_idx(nvshmemi_gic_device_dci_t *dci) {
    return READ_ONCE(dci->tx_wq.cons_head);
}

__device__ static inline void gic_update_current_cons_idx(nvshmemi_gic_device_dci_t *dci, uint64_t new_idx) {
    WRITE_ONCE(dci->tx_wq.cons_head, new_idx);
}

__device__ static inline void gic_update_current_get_idx(nvshmemi_gic_device_dci_t *dci, uint64_t new_idx) {
    WRITE_ONCE(dci->tx_wq.get_head, new_idx);
}

__device__ static inline gic_wqe_t *gic_get_wqe_ptr(nvshmemi_gic_device_dci_t *dci, uint16_t wqe_idx) {
    uint16_t cnt = dci->tx_wq.nwqes;
    uint16_t idx = wqe_idx & (cnt - 1); 
    return (gic_wqe_t *)((uintptr_t)dci->tx_wq.wqe + (idx << MLX5_SEND_WQE_SHIFT));
}

__device__ static void gic_write_rdma_write_wqe(
    nvshmemi_gic_device_dci_t *dci, nvshmemi_gic_device_dct_t *dct
    , uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey, uint32_t bytes
    , uint16_t wqe_idx, gic_wqe_t *out_wqe) {

    gic_wqe_dc_rdma_rw_t *wqe = &out_wqe->dc_rdma_write;
    
    wqe->raddr_seg.raddr = HTOBE64(raddr);
    wqe->raddr_seg.rkey = rkey;
    wqe->raddr_seg.reserved = 0;

    wqe->data_seg.byte_count = HTOBE32(bytes);
    wqe->data_seg.lkey = lkey;
    wqe->data_seg.addr = HTOBE64(laddr);

    wqe->av_seg = dci->half_av_seg_template;
    wqe->av_seg.dc_key = dct->access_key;
    wqe->av_seg.dqp_dct = dct->qpn;
    wqe->av_seg.rlid = dct->lid;

    wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(4)];
    wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(wqe_idx << 8 | MLX5_OPCODE_RDMA_WRITE);
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GIC_MAX_INLINE_SIZE <= 12, "NVSHMEMI_GIC_MAX_INLINE_SIZE <= 12 is mandatory.");
#endif
__device__ static void gic_write_rdma_write_inl_wqe(
    nvshmemi_gic_device_dci_t *dci, nvshmemi_gic_device_dct_t *dct
    , const void *val, uint64_t raddr, __be32 rkey, uint32_t bytes
    , uint16_t wqe_idx, gic_wqe_t *out_wqe) {

    gic_wqe_dc_rdma_write_inl_t *wqe = &out_wqe->dc_rdma_write_inl;
    void *wqe_data_ptr = (void *)((uintptr_t)wqe + sizeof(*wqe));

    assert(bytes <= NVSHMEMI_GIC_MAX_INLINE_SIZE);

    wqe->raddr_seg.raddr = HTOBE64(raddr);
    wqe->raddr_seg.rkey = rkey;
    wqe->raddr_seg.reserved = 0;

    wqe->inl_seg.byte_count = HTOBE32(bytes | MLX5_INLINE_SEG);
    memcpy(wqe_data_ptr, val, bytes);

    wqe->av_seg = dci->half_av_seg_template;
    wqe->av_seg.dc_key = dct->access_key;
    wqe->av_seg.dqp_dct = dct->qpn;
    wqe->av_seg.rlid = dct->lid;

    wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(4)];
    wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(wqe_idx << 8 | MLX5_OPCODE_RDMA_WRITE);
}

__device__ static void gic_write_rdma_read_wqe(
    nvshmemi_gic_device_dci_t *dci, nvshmemi_gic_device_dct_t *dct
    , uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey, uint32_t bytes
    , uint16_t wqe_idx, gic_wqe_t *out_wqe) {

    gic_wqe_dc_rdma_rw_t *wqe = &out_wqe->dc_rdma_read;
    
    wqe->raddr_seg.raddr = HTOBE64(raddr);
    wqe->raddr_seg.rkey = rkey;
    wqe->raddr_seg.reserved = 0;

    wqe->data_seg.byte_count = HTOBE32(bytes);
    wqe->data_seg.lkey = lkey;
    wqe->data_seg.addr = HTOBE64(laddr);

    wqe->av_seg = dci->half_av_seg_template;
    wqe->av_seg.dc_key = dct->access_key;
    wqe->av_seg.dqp_dct = dct->qpn;
    wqe->av_seg.rlid = dct->lid;

    wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(4)];
    wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(wqe_idx << 8 | MLX5_OPCODE_RDMA_READ);
}

__device__ static void gic_write_atomic_wqe(
    nvshmemi_gic_device_dci_t *dci, nvshmemi_gic_device_dct_t *dct
    , const void *val_1, const void *val_2, uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey, uint32_t bytes
    , uint16_t wqe_idx, nvshmemi_amo_t amo_op, gic_wqe_t *out_wqe_1, gic_wqe_t *out_wqe_2) {

    gic_wqe_dc_atomic_5ds_t *wqe = &out_wqe_1->dc_atomic_5ds;
    gic_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg = (gic_atomic_64_masked_cs_seg_t *)out_wqe_2;
    // The ptr is shifted if wqe is gic_wqe_dc_atomic_6ds_t
    struct mlx5_wqe_data_seg *data_seg = (struct mlx5_wqe_data_seg *)out_wqe_2;

    wqe->raddr_seg.raddr = HTOBE64(raddr);
    wqe->raddr_seg.rkey = rkey;
    wqe->raddr_seg.reserved = 0;

    wqe->av_seg = dci->half_av_seg_template;
    wqe->av_seg.dc_key = dct->access_key;
    wqe->av_seg.dqp_dct = dct->qpn;
    wqe->av_seg.rlid = dct->lid;

    // TODO: Check wrap around
    assert(bytes == 4 || bytes == 8);
    switch (amo_op) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC: {
            wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
            if (bytes == 4) {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_fa_seg.add_data = HTOBE32((uint32_t)1);
                wqe->atomic_32_masked_fa_seg.field_boundary = 0;
            } else {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_8_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_64_masked_fa_seg.add_data = HTOBE64((uint64_t)1);
                wqe->atomic_64_masked_fa_seg.field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET: {
            if (bytes == 4) {
                wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_cs_seg.swap_data = HTOBE32(*(uint32_t *)val_1);
                wqe->atomic_32_masked_cs_seg.compare_data = 0;
                wqe->atomic_32_masked_cs_seg.compare_mask = 0;
                wqe->atomic_32_masked_cs_seg.swap_mask = UINT32_MAX;
            } else {
                gic_wqe_dc_atomic_6ds_t *wqe = &out_wqe_1->dc_atomic_6ds;
                wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(6)];
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) | GIC_8_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_64_masked_cs_data_seg.swap = HTOBE64(*(uint64_t *)val_1);
                wqe->atomic_64_masked_cs_data_seg.compare = 0;
                atomic_64_masked_cs_mask_seg->swap = UINT64_MAX;
                atomic_64_masked_cs_mask_seg->compare = 0;
                data_seg = (struct mlx5_wqe_data_seg *)((uintptr_t)out_wqe_2 + sizeof(gic_atomic_64_masked_cs_seg_t));
            }
            break;
        }
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
            if (bytes == 4) {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_fa_seg.add_data = HTOBE32(*(uint32_t *)val_1);
                wqe->atomic_32_masked_fa_seg.field_boundary = 0;
            } else {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_8_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_64_masked_fa_seg.add_data = HTOBE64(*(uint64_t *)val_1);
                wqe->atomic_64_masked_fa_seg.field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            if (bytes == 4) {
                wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_cs_seg.swap_data = HTOBE32(*(uint32_t *)val_1);
                wqe->atomic_32_masked_cs_seg.compare_data = 0;
                wqe->atomic_32_masked_cs_seg.compare_mask = 0;
                wqe->atomic_32_masked_cs_seg.swap_mask = HTOBE32(~(*(uint32_t *)val_1));
            } else {
                gic_wqe_dc_atomic_6ds_t *wqe = &out_wqe_1->dc_atomic_6ds;
                wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(6)];
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) | GIC_8_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_64_masked_cs_data_seg.swap = HTOBE64(*(uint64_t *)val_1);
                wqe->atomic_64_masked_cs_data_seg.compare = 0;
                atomic_64_masked_cs_mask_seg->swap = HTOBE64(~(*(uint64_t *)val_1));
                atomic_64_masked_cs_mask_seg->compare = 0;
                data_seg = (struct mlx5_wqe_data_seg *)((uintptr_t)out_wqe_2 + sizeof(gic_atomic_64_masked_cs_seg_t));
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            if (bytes == 4) {
                wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_cs_seg.swap_data = HTOBE32(*(uint32_t *)val_1);
                wqe->atomic_32_masked_cs_seg.compare_data = 0;
                wqe->atomic_32_masked_cs_seg.compare_mask = 0;
                wqe->atomic_32_masked_cs_seg.swap_mask = HTOBE32(*(uint32_t *)val_1);
            } else {
                gic_wqe_dc_atomic_6ds_t *wqe = &out_wqe_1->dc_atomic_6ds;
                wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(6)];
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) | GIC_8_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_64_masked_cs_data_seg.swap = HTOBE64(*(uint64_t *)val_1);
                wqe->atomic_64_masked_cs_data_seg.compare = 0;
                atomic_64_masked_cs_mask_seg->swap = HTOBE64(*(uint64_t *)val_1);
                atomic_64_masked_cs_mask_seg->compare = 0;
                data_seg = (struct mlx5_wqe_data_seg *)((uintptr_t)out_wqe_2 + sizeof(gic_atomic_64_masked_cs_seg_t));
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
            if (bytes == 4) {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_fa_seg.add_data = HTOBE32(*(uint32_t *)val_1);
                wqe->atomic_32_masked_fa_seg.field_boundary = UINT32_MAX;
            } else {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_8_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_64_masked_fa_seg.add_data = HTOBE64(*(uint64_t *)val_1);
                wqe->atomic_64_masked_fa_seg.field_boundary = UINT64_MAX;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
            if (bytes == 4) {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_fa_seg.add_data = 0;
                wqe->atomic_32_masked_fa_seg.field_boundary = 0;
            } else {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_8_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_64_masked_fa_seg.add_data = 0;
                wqe->atomic_64_masked_fa_seg.field_boundary = 0;
            }
            break;
        }
        case NVSHMEMI_AMO_FETCH_ADD: {
            wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
            if (bytes == 4) {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_fa_seg.add_data = HTOBE32(*(uint32_t *)val_1);
                wqe->atomic_32_masked_fa_seg.field_boundary = 0;
            } else {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_FA | (wqe_idx << 8));
                wqe->atomic_seg.swap_add = HTOBE64(*(uint64_t *)val_1);
            }
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(5)];
            if (bytes == 4) {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_MASKED_CS | (wqe_idx << 8) | GIC_4_BYTE_EXT_AMO_OPMOD);
                wqe->atomic_32_masked_cs_seg.swap_data = HTOBE32(*(uint32_t *)val_1);
                wqe->atomic_32_masked_cs_seg.compare_data = HTOBE32(*(uint32_t *)val_2);
                wqe->atomic_32_masked_cs_seg.compare_mask = UINT32_MAX;
                wqe->atomic_32_masked_cs_seg.swap_mask = UINT32_MAX;
            } else {
                wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(MLX5_OPCODE_ATOMIC_CS | (wqe_idx << 8));
                wqe->atomic_seg.swap_add = HTOBE64(*(uint64_t *)val_1);
                wqe->atomic_seg.compare = HTOBE64(*(uint64_t *)val_2);
            }
            break;
        }
        default: {
            assert(0);
        }
    }

    data_seg->byte_count = HTOBE32(bytes);
    data_seg->lkey = lkey;
    data_seg->addr = HTOBE64(laddr);
}

__device__ static void gic_write_dump_wqe(
    nvshmemi_gic_device_dci_t *dci, nvshmemi_gic_device_dct_t *dct
    , uint64_t laddr, __be32 lkey, uint32_t bytes
    , uint16_t wqe_idx, gic_wqe_t *out_wqe) {

    gic_wqe_dc_dump_t *wqe = &out_wqe->dc_dump;
    
    wqe->data_seg.byte_count = HTOBE32(bytes);
    wqe->data_seg.lkey = lkey;
    wqe->data_seg.addr = HTOBE64(laddr);

    wqe->ctrl_seg = dci->ctrl_seg_templates[nvshmemi_gic_ctrl_seg_ds_to_template_idx(2)];
    wqe->ctrl_seg.opmod_idx_opcode = HTOBE32(wqe_idx << 8 | GIC_MLX5_OPCODE_DUMP);
}

__device__ static void gic_update_dbr(nvshmemi_gic_device_dci_t *dci, uint32_t dbrec_head) {
    // DBREC contains the index of the next empty WQEBB.
    __be32 dbrec_val;
    __be32 *dbrec_ptr = dci->tx_wq.dbrec;

    // This is equivalent to 
    // WRITE_ONCE(dbrec_ptr, HTOBE32(dbrec_head & 0xffff));
    asm volatile("{\n\t"
        ".reg .b32 mask1;\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 mask2;\n\t"
        "mov.b32 mask1, 0xffff;\n\t"
        "mov.b32 mask2, 0x123;\n\t"
        "and.b32 dbrec_head_16b, %1, mask1;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, mask2;\n\t"
        "}" : "=r" (dbrec_val) : "r" (dbrec_head));
    WRITE_ONCE(*dbrec_ptr, dbrec_val);
}

__device__ static void gic_ring_db(nvshmemi_gic_device_dci_t *dci, gic_wqe_t *wqe) {
    uint64_t *bf_ptr = (uint64_t *)dci->tx_wq.bf;

    WRITE_ONCE(*bf_ptr, *(uint64_t *)wqe);
}

template <bool must_lock>
__device__ static inline uint64_t gic_cst(nvshmemi_gic_device_dci_t *dci) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, nvshmemi_device_state_d.mype);

    if (must_lock)
        gic_lock<NVSHMEMI_THREADGROUP_THREAD>(&dci->lock);

    uint64_t laddr = (uint64_t)dci->internal_buf.buf;
    __be32 lkey = dci->internal_buf.lkey;

    uint64_t raddr = laddr;
    __be32 rkey = dci->internal_buf.rkey;

    uint64_t wqe_idx = gic_get_current_wqe_idx(dci);
    uint64_t cons_idx = gic_get_current_cons_idx(dci);
    gic_wqe_t *wqe_ptr = gic_get_wqe_ptr(dci, wqe_idx);

    gic_ensure_wqe_availability(dci, 1);
    if (state->nic_buf_on_gpumem)
        // For CST, DUMP OP is cheaper than RDMA READ.
        // However, it works only if WQ buffer is on GPU memory.
        gic_write_dump_wqe(dci, dct, laddr, lkey, sizeof(char), wqe_idx, wqe_ptr);
    else
        gic_write_rdma_read_wqe(dci, dct, laddr, lkey, raddr, rkey, sizeof(char), wqe_idx, wqe_ptr);

    // Don't update get_head here because this is internal cst
    GIC_MFENCE();
    gic_update_dbr(dci, wqe_idx + 1);
    gic_update_current_wqe_idx(dci, wqe_idx + 1);
    gic_update_current_cons_idx(dci, cons_idx + 1);
    GIC_MFENCE();
    gic_ring_db(dci, wqe_ptr);

    if (must_lock)
        gic_unlock<NVSHMEMI_THREADGROUP_THREAD>(&dci->lock);

    return gic_quiet(dci);
}

template <bool must_lock>
__device__ static uint64_t gic_quiet_with_cst(nvshmemi_gic_device_dci_t *dci) {
    uint64_t head = READ_ONCE(dci->tx_wq.get_head);
    uint64_t ticket = gic_quiet(dci);
    uint64_t tail = READ_ONCE(dci->tx_wq.get_tail);

    if (tail < head) {
        ticket = gic_cst<must_lock>(dci);
        atomicMax((unsigned long long int *)&dci->tx_wq.get_tail, (unsigned long long int)ticket);
    }

    return ticket;
}

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op, bool nbi>
__device__ static inline void gic_rma(void *rptr, void *lptr, size_t bytes, int pe) {
    constexpr bool need_cst = !!(channel_op == NVSHMEMI_OP_GET);

    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 lkey = gic_get_lkey(state, (uint64_t)lptr);

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci = gic_get_dci<SCOPE>(state);

    if (my_tid == 0) {
        uint64_t wqe_idx = gic_get_current_wqe_idx(dci);
        uint64_t cons_idx = gic_get_current_cons_idx(dci);
        gic_wqe_t *wqe_ptr = gic_get_wqe_ptr(dci, wqe_idx);
        gic_ensure_wqe_availability(dci, 1);
        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                gic_write_rdma_write_wqe(dci, dct, (uint64_t)lptr, lkey, raddr, rkey, bytes, wqe_idx, wqe_ptr);
                break;
            case NVSHMEMI_OP_GET:
                gic_write_rdma_read_wqe(dci, dct, (uint64_t)lptr, lkey, raddr, rkey, bytes, wqe_idx, wqe_ptr);
                // GET index must be visible before the new cons index.
                gic_update_current_get_idx(dci, cons_idx + 1);
                break;
            default:
                #ifdef NVSHMEM_GPUINITIATED_DEBUG
                printf("Unsupported channel_op.\n");
                #endif
                assert(0);
        }
        GIC_MFENCE();
        gic_update_dbr(dci, wqe_idx + 1);
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        gic_update_current_wqe_idx(dci, wqe_idx + 1);
        gic_update_current_cons_idx(dci, cons_idx + 1);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr);

        if (!nbi && need_cst)
            gic_quiet_with_cst<false>(dci);
    }

    gic_release_dci<SCOPE>(dci);

    // Release this dci so that others can use it.
    // Then, we can do quiet, which doesn't require locking.
    if (!nbi && my_tid == 0 && !need_cst)
        gic_quiet(dci);

    nvshmemi_threadgroup_sync<SCOPE>();
}

template <nvshmemi_op_t channel_op, bool nbi>
__device__ static inline void gic_rma_thread(void *rptr, void *lptr, size_t bytes, int pe) {
    constexpr bool need_cst = !!(channel_op == NVSHMEMI_OP_GET);

    unsigned int amask = __activemask();

    int tid;
    int tg_size;
    uint16_t num_required_wqes;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 lkey = gic_get_lkey(state, (uint64_t)lptr);

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci;

    if (amask == GIC_FULL_WARP) {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_WARP>(state);
    } else {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_THREAD>(state);
    }
    num_required_wqes = tg_size;

    uint64_t wqe_idx = gic_get_current_wqe_idx(dci);
    uint64_t cons_idx = gic_get_current_cons_idx(dci);

    gic_wqe_t *wqe_ptr = gic_get_wqe_ptr(dci, wqe_idx + tid);

    if (tid == 0)
        gic_ensure_wqe_availability(dci, tg_size);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    switch (channel_op) {
        case NVSHMEMI_OP_PUT:
            gic_write_rdma_write_wqe(dci, dct, (uint64_t)lptr, lkey, raddr, rkey, bytes, wqe_idx + tid, wqe_ptr);
            break;
        case NVSHMEMI_OP_GET:
            gic_write_rdma_read_wqe(dci, dct, (uint64_t)lptr, lkey, raddr, rkey, bytes, wqe_idx + tid, wqe_ptr);
            break;
        default:
            #ifdef NVSHMEM_GPUINITIATED_DEBUG
            printf("Unsupported channel_op.\n");
            #endif
            assert(0);
    }

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    if (tid == tg_size - 1) {
        if (need_cst) {
            // GET index must be visible before the new cons index.
            gic_update_current_get_idx(dci, cons_idx + tg_size);
        }

        GIC_MFENCE();
        gic_update_dbr(dci, wqe_idx + num_required_wqes);
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        gic_update_current_wqe_idx(dci, wqe_idx + num_required_wqes);
        gic_update_current_cons_idx(dci, cons_idx + tg_size);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr);

        if (!nbi && need_cst)
            gic_quiet_with_cst<false>(dci);
    }

    if (amask == GIC_FULL_WARP)
        gic_release_dci<NVSHMEMI_THREADGROUP_WARP>(dci);
    else
        gic_release_dci<NVSHMEMI_THREADGROUP_THREAD>(dci);

    // Release this dci so that others can use it.
    // Then, we can do quiet, which doesn't require locking.
    if (!nbi && tid == 0 && !need_cst)
        gic_quiet(dci);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();
}

/**
 * RMA P base
 */
template <typename T>
__device__ void nvshmemi_gic_rma_p(void *rptr, const T value, int pe) {
    unsigned int amask = __activemask();
    int tid;
    int tg_size;
    uint16_t num_required_wqes;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci;

    if (amask == GIC_FULL_WARP) {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_WARP>(state);
    } else {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_THREAD>(state);
    }
    num_required_wqes = tg_size;

    uint64_t wqe_idx = gic_get_current_wqe_idx(dci);
    uint64_t cons_idx = gic_get_current_cons_idx(dci);

    gic_wqe_t *wqe_ptr = gic_get_wqe_ptr(dci, wqe_idx + tid);

    if (tid == 0)
        gic_ensure_wqe_availability(dci, tg_size);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    gic_write_rdma_write_inl_wqe(dci, dct, &value, raddr, rkey, sizeof(T), wqe_idx + tid, wqe_ptr);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    if (tid == tg_size - 1) {
        GIC_MFENCE();
        gic_update_dbr(dci, wqe_idx + num_required_wqes);
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        gic_update_current_wqe_idx(dci, wqe_idx + num_required_wqes);
        gic_update_current_cons_idx(dci, cons_idx + tg_size);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr);
    }

    if (amask == GIC_FULL_WARP)
        gic_release_dci<NVSHMEMI_THREADGROUP_WARP>(dci);
    else
        gic_release_dci<NVSHMEMI_THREADGROUP_THREAD>(dci);
}

/**
 * RMA P
 */
#define GIC_DECL_RMA_P(Type) \
template __device__ void nvshmemi_gic_rma_p<Type>(void *rptr, const Type value, int pe);

GIC_REPT_FOR_STANDARD_RMA_TYPES(GIC_DECL_RMA_P)

/**
 * RMA G base
 */
template<typename T>
__device__ T nvshmemi_gic_rma_g(void *rptr, int pe) {
    unsigned int amask = __activemask();
    int tid;
    int tg_size;
    uint16_t num_required_wqes;

    T ret;
    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci;

    if (amask == GIC_FULL_WARP) {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_WARP>(state);
    } else {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_THREAD>(state);
    }
    num_required_wqes = tg_size;

    __be32 lkey = dci->internal_buf.lkey;
    uint64_t laddr = (uint64_t)dci->internal_buf.buf + sizeof(uint64_t) + (sizeof(T) * tid);

    uint64_t wqe_idx = gic_get_current_wqe_idx(dci);
    uint64_t cons_idx = gic_get_current_cons_idx(dci);

    gic_wqe_t *wqe_ptr = gic_get_wqe_ptr(dci, wqe_idx + tid);

    if (tid == 0)
        gic_ensure_wqe_availability(dci, tg_size);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    gic_write_rdma_read_wqe(dci, dct, laddr, lkey, raddr, rkey, sizeof(T), wqe_idx + tid, wqe_ptr);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    if (tid == tg_size - 1) {
        // GET index must be visible before the new cons index.
        gic_update_current_get_idx(dci, cons_idx + tg_size);
        GIC_MFENCE();
        gic_update_dbr(dci, wqe_idx + num_required_wqes);
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        gic_update_current_wqe_idx(dci, wqe_idx + num_required_wqes);
        gic_update_current_cons_idx(dci, cons_idx + tg_size);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr);

        gic_quiet_with_cst<false>(dci);
    }

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);

    if (amask == GIC_FULL_WARP)
        gic_release_dci<NVSHMEMI_THREADGROUP_WARP>(dci);
    else
        gic_release_dci<NVSHMEMI_THREADGROUP_THREAD>(dci);

    return ret;
}

/**
 * RMA G
 */
#define GIC_DECL_RMA_G(Type) \
template __device__ Type nvshmemi_gic_rma_g<Type>(void *rptr, int pe);

GIC_REPT_FOR_STANDARD_RMA_TYPES(GIC_DECL_RMA_G)

/**
 * RMA NBI base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ void nvshmemi_gic_rma_nbi(void *rptr, void *lptr, size_t bytes, int pe) {
    gic_rma<SCOPE, channel_op, true>(rptr, lptr, bytes, pe);
}

/**
 * RMA NBI PUT/GET
 */
#define GIC_DECL_RMA_NBI(SCOPE) \
template __device__ void nvshmemi_gic_rma_nbi \
    <SCOPE, NVSHMEMI_OP_PUT>(void *rptr, void *lptr, size_t bytes, int pe); \
template __device__ void nvshmemi_gic_rma_nbi \
    <SCOPE, NVSHMEMI_OP_GET>(void *rptr, void *lptr, size_t bytes, int pe);

GIC_REPT_FOR_ALL_SCOPES(GIC_DECL_RMA_NBI)


/**
 * RMA (blocking) base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ void nvshmemi_gic_rma(void *rptr, void *lptr, size_t bytes, int pe) {
    if (SCOPE == NVSHMEMI_THREADGROUP_BLOCK)
        gic_rma_thread<channel_op, false>(rptr, lptr, bytes, pe);
    else
        gic_rma<SCOPE, channel_op, false>(rptr, lptr, bytes, pe);
}

/**
 * RMA (blocking) PUT/GET
 */
#define GIC_DECL_RMA(SCOPE) \
template __device__ void nvshmemi_gic_rma \
    <SCOPE, NVSHMEMI_OP_PUT>(void *rptr, void *lptr, size_t bytes, int pe); \
template __device__ void nvshmemi_gic_rma \
    <SCOPE, NVSHMEMI_OP_GET>(void *rptr, void *lptr, size_t bytes, int pe);

GIC_REPT_FOR_ALL_SCOPES(GIC_DECL_RMA)

/**
 * AMO non-fetch base
 */
template <typename T>
__device__ void nvshmemi_gic_amo_nonfetch(void *rptr, const T value, int pe, nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int tid;
    int tg_size;
    uint16_t num_required_wqes;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci;

    if (amask == GIC_FULL_WARP) {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_WARP>(state);
    } else {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_THREAD>(state);
    }
    num_required_wqes = tg_size * 2;

    uint64_t base_wqe_idx = gic_get_current_wqe_idx(dci);
    uint64_t cons_idx = gic_get_current_cons_idx(dci);
    uint64_t my_wqe_idx = base_wqe_idx + (tid * 2);

    gic_wqe_t *wqe_ptr_1 = gic_get_wqe_ptr(dci, my_wqe_idx);
    gic_wqe_t *wqe_ptr_2 = gic_get_wqe_ptr(dci, my_wqe_idx + 1);

    if (tid == 0)
        gic_ensure_wqe_availability(dci, tg_size);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    gic_write_atomic_wqe(
        dci, dct, &value, NULL, (uint64_t)dci->internal_buf.buf, dci->internal_buf.lkey,
        raddr, rkey, sizeof(T), my_wqe_idx, op, wqe_ptr_1, wqe_ptr_2
    );

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    if (tid == tg_size - 1) {
        GIC_MFENCE();
        gic_update_dbr(dci, base_wqe_idx + num_required_wqes);
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        // Consume 2 WQEBB
        gic_update_current_wqe_idx(dci, base_wqe_idx + num_required_wqes);
        gic_update_current_cons_idx(dci, cons_idx + tg_size);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr_1);
    }

    if (amask == GIC_FULL_WARP)
        gic_release_dci<NVSHMEMI_THREADGROUP_WARP>(dci);
    else
        gic_release_dci<NVSHMEMI_THREADGROUP_THREAD>(dci);
}

/**
 * AMO non-fetch
 */
#define GIC_DECL_AMO_NONFETCH(Type) \
template __device__ void nvshmemi_gic_amo_nonfetch<Type>(void *rptr, const Type value, int pe, nvshmemi_amo_t op);

GIC_REPT_FOR_STANDARD_AMO_TYPES(GIC_DECL_AMO_NONFETCH);
GIC_REPT_FOR_EXTENDED_AMO_TYPES(GIC_DECL_AMO_NONFETCH);

/**
 * AMO fetch base
 */
template <typename T>
__device__ T nvshmemi_gic_amo_fetch(void *rptr, const T value, const T compare, int pe, nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int tid;
    int tg_size;
    uint16_t num_required_wqes;

    T ret;
    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci;

    if (amask == GIC_FULL_WARP) {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_WARP>(state);
    } else {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_THREAD>(state);
    }
    num_required_wqes = tg_size * 2;

    __be32 lkey = dci->internal_buf.lkey;
    uint64_t laddr = (uint64_t)dci->internal_buf.buf + sizeof(uint64_t) + (sizeof(T) * tid);

    uint64_t base_wqe_idx = gic_get_current_wqe_idx(dci);
    uint64_t cons_idx = gic_get_current_cons_idx(dci);
    uint64_t my_wqe_idx = base_wqe_idx + (tid * 2);
    gic_wqe_t *wqe_ptr_1 = gic_get_wqe_ptr(dci, my_wqe_idx);
    gic_wqe_t *wqe_ptr_2 = gic_get_wqe_ptr(dci, my_wqe_idx + 1);

    if (tid == 0)
        gic_ensure_wqe_availability(dci, tg_size);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    gic_write_atomic_wqe(
        dci, dct, &value, &compare, laddr, lkey,
        raddr, rkey, sizeof(T), my_wqe_idx, op, wqe_ptr_1, wqe_ptr_2
    );

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    if (tid == tg_size - 1) {
        // GET index must be visible before the new cons index.
        gic_update_current_get_idx(dci, cons_idx + tg_size);
        GIC_MFENCE();
        gic_update_dbr(dci, base_wqe_idx + num_required_wqes);
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        // Consume 2 WQEBB
        gic_update_current_wqe_idx(dci, base_wqe_idx + num_required_wqes);
        gic_update_current_cons_idx(dci, cons_idx + tg_size);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr_1);

        gic_quiet_with_cst<false>(dci);
    }

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);
    if (sizeof(T) == 4)
        ret = BSWAP32((uint32_t)ret);

    if (amask == GIC_FULL_WARP)
        gic_release_dci<NVSHMEMI_THREADGROUP_WARP>(dci);
    else
        gic_release_dci<NVSHMEMI_THREADGROUP_THREAD>(dci);

    return ret;
}

/**
 * AMO fetch base
 */
#define GIC_DECL_AMO_FETCH(Type) \
template __device__ Type nvshmemi_gic_amo_fetch<Type>(void *rptr, const Type value, const Type compare, int pe, nvshmemi_amo_t op);

GIC_REPT_FOR_STANDARD_AMO_TYPES(GIC_DECL_AMO_FETCH);
GIC_REPT_FOR_EXTENDED_AMO_TYPES(GIC_DECL_AMO_FETCH);

/**
 * PUT SIGNAL base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_GIC_MIN_NUM_BATCH_SIZE >= 2);
#endif
template <threadgroup_t SCOPE>
__device__ void nvshmemi_gic_put_signal(
    void *rptr, void *lptr, size_t bytes
    , void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op
    , int pe, bool is_nbi
) {
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 lkey = gic_get_lkey(state, (uint64_t)lptr);

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    __be32 sig_rkey;
    uint64_t sig_raddr;
    gic_get_raddr_rkey(state, (uint64_t)sig_rptr, pe, &sig_raddr, &sig_rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci = gic_get_dci<SCOPE>(state);

    if (my_tid == 0) {
        uint64_t wqe_idx = gic_get_current_wqe_idx(dci);
        uint64_t cons_idx = gic_get_current_cons_idx(dci);
        gic_wqe_t *wqe_ptr = gic_get_wqe_ptr(dci, wqe_idx);
        gic_ensure_wqe_availability(dci, 2);
        gic_write_rdma_write_wqe(dci, dct, (uint64_t)lptr, lkey, raddr, rkey, bytes, wqe_idx, wqe_ptr);

        gic_wqe_t *wqe_ptr_1 = gic_get_wqe_ptr(dci, wqe_idx + 1);
        gic_wqe_t *wqe_ptr_2 = gic_get_wqe_ptr(dci, wqe_idx + 2);

        gic_write_atomic_wqe(
            dci, dct, &signal, NULL, (uint64_t)dci->internal_buf.buf, dci->internal_buf.lkey,
            sig_raddr, sig_rkey, sizeof(signal), wqe_idx + 1, sig_op, wqe_ptr_1, wqe_ptr_2
        );

        GIC_MFENCE();
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        // Consume 2 WQEBB
        gic_update_current_wqe_idx(dci, wqe_idx + 3);
        gic_update_current_cons_idx(dci, cons_idx + 2);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr_1);
    }

    gic_release_dci<SCOPE>(dci);

    if (is_nbi == 0)
        gic_quiet(dci);
}

template <>
__device__ void nvshmemi_gic_put_signal<NVSHMEMI_THREADGROUP_THREAD>(
    void *rptr, void *lptr, size_t bytes
    , void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op
    , int pe, bool is_nbi
) {
    unsigned int amask = __activemask();

    int tid;
    int tg_size;
    uint16_t num_required_wqes;

    nvshmemi_gic_device_state_t *state = gic_get_state();

    __be32 lkey = gic_get_lkey(state, (uint64_t)lptr);

    __be32 rkey;
    uint64_t raddr;
    gic_get_raddr_rkey(state, (uint64_t)rptr, pe, &raddr, &rkey);

    __be32 sig_rkey;
    uint64_t sig_raddr;
    gic_get_raddr_rkey(state, (uint64_t)sig_rptr, pe, &sig_raddr, &sig_rkey);

    nvshmemi_gic_device_dct_t *dct = gic_get_dct(state, pe);
    nvshmemi_gic_device_dci_t *dci;

    if (amask == GIC_FULL_WARP) {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_WARP>(state);
    } else {
        tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        dci = gic_get_dci<NVSHMEMI_THREADGROUP_THREAD>(state);
    }
    // 3 WQEBBs per thread are required.
    num_required_wqes = tg_size * 3;

    uint64_t base_wqe_idx = gic_get_current_wqe_idx(dci);
    uint64_t my_wqe_idx = base_wqe_idx + tid * 3;
    uint64_t cons_idx = gic_get_current_cons_idx(dci);

    gic_wqe_t *wqe_ptr = gic_get_wqe_ptr(dci, my_wqe_idx);
    gic_wqe_t *wqe_ptr_1 = gic_get_wqe_ptr(dci, my_wqe_idx + 1);
    gic_wqe_t *wqe_ptr_2 = gic_get_wqe_ptr(dci, my_wqe_idx + 2);

    if (tid == 0)
        gic_ensure_wqe_availability(dci, tg_size * 2); // We will advance 2 cons_idx per thread

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    gic_write_rdma_write_wqe(dci, dct, (uint64_t)lptr, lkey, raddr, rkey, bytes, my_wqe_idx, wqe_ptr);

    gic_write_atomic_wqe(
        dci, dct, &signal, NULL, (uint64_t)dci->internal_buf.buf, dci->internal_buf.lkey,
        sig_raddr, sig_rkey, sizeof(signal), my_wqe_idx + 1, sig_op, wqe_ptr_1, wqe_ptr_2
    );

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();

    if (tid == tg_size - 1) {
        GIC_MFENCE();
        gic_update_dbr(dci, base_wqe_idx + num_required_wqes);
        // Update wqe idx before ringing the db so that we know which index is needed in quiet/fence.
        gic_update_current_wqe_idx(dci, base_wqe_idx + num_required_wqes);
        gic_update_current_cons_idx(dci, cons_idx + tg_size * 2);
        GIC_MFENCE();
        gic_ring_db(dci, wqe_ptr_1);
    }

    if (amask == GIC_FULL_WARP)
        gic_release_dci<NVSHMEMI_THREADGROUP_WARP>(dci);
    else
        gic_release_dci<NVSHMEMI_THREADGROUP_THREAD>(dci);

    if (is_nbi == 0 && tid == 0)
        gic_quiet(dci);

    if (amask == GIC_FULL_WARP)
        nvshmemi_warp_sync();
}

/**
 * PUT SIGNAL
 */
#define GIC_DECL_PUT_SIGNAL(SCOPE) \
template __device__ void nvshmemi_gic_put_signal \
    <SCOPE>(void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op, int pe, bool is_nbi);

GIC_REPT_FOR_ALL_SCOPES(GIC_DECL_PUT_SIGNAL)


__device__ void nvshmemi_gic_quiet() {
    unsigned int amask = __activemask();
    nvshmemi_gic_device_state_t *state = gic_get_state();

    if (amask == GIC_FULL_WARP) {
        for (uint32_t i = nvshmemi_thread_id_in_warp(); i < state->ndcis; i += warpSize)
            gic_quiet_with_cst<true>(&state->dcis[i]);
        nvshmemi_warp_sync();
    } else {
        for (uint32_t i = 0; i < state->ndcis; ++i)
            gic_quiet_with_cst<true>(&state->dcis[i]);
    }
}

__device__ void nvshmemi_gic_fence() {
    // Multiple DCIs may target the same PE before fence.
    // We need to quiet those DCIs.
    // TODO: Make it more efficient.
    nvshmemi_gic_quiet();
}

__device__ void nvshmemi_gic_enforce_consistency_at_target(bool use_membar) {
    nvshmemi_gic_device_state_t *state = gic_get_state();
    nvshmemi_gic_device_dci_t *dci = gic_get_dci<NVSHMEMI_THREADGROUP_THREAD>(state);
    gic_cst<false>(dci);
    gic_release_dci<NVSHMEMI_THREADGROUP_THREAD>(dci);

    // TODO: This fence is from the design of Proxy.
    // Review if we still need it when we fully move to GIC -- especially for on-stream API.
    if (use_membar) {
        __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                                // cst_ack_d (breaks cst -> rma)
    }
}

#endif /* __CUDA_ARCH__ */
