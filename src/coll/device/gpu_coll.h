/*
 * * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 * *
 * * See COPYRIGHT for license information
 * */

#ifndef _NVSHMEMI_COLL_GPU_H_
#define _NVSHMEMI_COLL_GPU_H_

#include "nvshmem_internal.h"
#include "common.cuh"
#include "coll_shorthand.h"

#define WITH_QUIET 1
#define WITHOUT_QUIET 0
#define WITH_CST 1
#define WITHOUT_CST 0


/* structs */

typedef struct gpu_coll_env_params {
    int gpu_intm_rdxn_size;
    int reduce_recexch_kval;
} gpu_coll_env_params_t;

typedef struct gpu_coll_info {
    volatile long *ipsync;
    volatile int4 *ipwrk;
    volatile long *icounter;
    volatile long *icounter_barrier;
    void *ipc_shm_addr;
    void *own_shm_addr;
    volatile char *own_intm_addr;
    volatile char *own_intm_rdxn_addr;
} gpu_coll_info_t;

typedef enum gpu_rdxn_op {
    gpu_rd_and = 0,
    gpu_rd_max,
    gpu_rd_min,
    gpu_rd_sum,
    gpu_rd_prod,
    gpu_rd_or,
    gpu_rd_xor,
    gpu_rd_null
} gpu_rdxn_op_t;

typedef enum gpu_rdxn_op_dt {
    gpu_rd_dt_short = 0,
    gpu_rd_dt_int,
    gpu_rd_dt_long,
    gpu_rd_dt_float,
    gpu_rd_dt_double,
    gpu_rd_dt_long_long,
    gpu_rd_dt_long_double,
    gpu_rd_dt_float_complex,
    gpu_rd_dt_double_complex,
    gpu_rd_dt_null
} gpu_rdxn_op_dt_t;

typedef enum gpu_bits_opr_dt {
    gpu_bits_dt_8 = 0,
    gpu_bits_dt_16,
    gpu_bits_dt_32,
    gpu_bits_dt_64,
    gpu_bits_dt_null
} gpu_bits_opr_dt_t;

/* global vars */
extern gpu_coll_info_t nvshm_gpu_coll_info;
extern int nvshm_gpu_coll_initialized;
extern gpu_coll_env_params_t gpu_coll_env_params_var;

extern __device__ volatile int *gpu_bcast_int_sync_arr_d;
extern __device__ volatile int *gpu_bcast_int_data_arr_d;
extern __device__ volatile char *gpu_own_intm_addr_d;
extern __device__ volatile char *gpu_own_intm_rdxn_addr_d;
extern __device__ volatile long *gpu_ipsync_d;
extern __device__ volatile int4 *gpu_ipwrk_d;
extern __device__ long *gpu_icounter_d;
extern __device__ long *gpu_icounter_barrier_d;
extern __device__ gpu_coll_env_params_t gpu_coll_env_params_var_d;

extern __device__ int reduce_recexch_step1_sendto_d;
extern __device__ int *reduce_recexch_step1_recvfrom_d;
extern __device__ int reduce_recexch_step1_nrecvs_d;
extern __device__ int **reduce_recexch_step2_nbrs_d;
extern __device__ int reduce_recexch_step2_nphases_d;
extern __device__ int reduce_recexch_p_of_k_d;
extern __device__ int reduce_recexch_reduce_recexch_digit_d;
extern __device__ int *digit_d;


/* function declarations */
extern int nvshmemi_coll_common_gpu_init(void);
extern int nvshmemi_coll_common_gpu_return_modes(void);
extern int nvshmemi_coll_common_gpu_finalize(void);
extern int nvshmemi_alltoall_gpu_init(void);
extern int nvshmemi_alltoall_gpu_finalize(void);
extern int nvshmemi_alltoalls_gpu_init(void);
extern int nvshmemi_alltoalls_gpu_finalize(void);
extern int nvshmemi_barrier_gpu_init(void);
extern int nvshmemi_barrier_gpu_finalize(void);
extern int nvshmemi_barrier_all_gpu_init(void);
extern int nvshmemi_barrier_all_gpu_finalize(void);
extern int nvshmemi_broadcast_gpu_init(void);
extern int nvshmemi_broadcast_gpu_finalize(void);
extern int nvshmemi_collect_gpu_init(void);
extern int nvshmemi_collect_gpu_finalize(void);
extern int nvshmemi_fcollect_gpu_init(void);
extern int nvshmemi_fcollect_gpu_finalize(void);
extern int nvshmemi_reduction_gpu_init(void);
extern int nvshmemi_reduction_gpu_finalize(void);

__device__ void nvshmemi_collect(void *dest, const void *source, int type_size,
                                 gpu_bits_opr_dt_t bits_dt, size_t nelems, int PE_start,
                                 int logPE_stride, int PE_size, long *pSync);

void nvshmemi_recexchalgo_get_neighbors(int my_pe, int PE_size);

#if __cplusplus
extern "C" {
#endif
int init_shm_kernel_shm_ptr(gpu_coll_info_t *nvshm_gpu_coll_info);
#if __cplusplus
}
#endif

/* macro definitions */
#define NVSHMEMI_COLL_GPU_STATUS_SUCCESS 0
#define NVSHMEMI_COLL_GPU_STATUS_ERROR 1
#define NVSHMEMI_COLL_GPU_SHM_BLK_SIZE 16384
#define MAX_THREADS_PER_CTA 1024

#define NVSHMEMI_COLL_GPU_ERR_POP()                                                \
    do {                                                                           \
        fprintf(stderr, "Error at %s:%d in %s", __FILE__, __LINE__, __FUNCTION__); \
        goto fn_fail;                                                              \
    } while (0)

#endif /* NVSHMEMI_COLL_GPU_H */
