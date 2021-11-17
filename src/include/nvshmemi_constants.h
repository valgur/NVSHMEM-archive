/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMI_CONSTANTS_H_
#define _NVSHMEMI_CONSTANTS_H_

#define NVSHMEMI_DIAG_STRLEN 1024

#define SYNC_SIZE 27648 /*XXX:Number of GPUs on Summit; currently O(N), need to be O(1)*/
#define NVSHMEMI_SYNC_VALUE 0
#define NVSHMEMI_SYNC_SIZE (2 * SYNC_SIZE)
#define NVSHMEMI_BARRIER_SYNC_SIZE (2 * SYNC_SIZE)
#define NVSHMEMI_BCAST_SYNC_SIZE SYNC_SIZE
#define NVSHMEMI_REDUCE_SYNC_SIZE SYNC_SIZE
#define NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE SYNC_SIZE
#define NVSHMEMI_COLLECT_SYNC_SIZE SYNC_SIZE
#define NVSHMEMI_ALLTOALL_SYNC_SIZE SYNC_SIZE

#define NVSHMEMI_WARP_SIZE 32

typedef enum {
    NVSHMEMI_OP_PUT = 1,
    NVSHMEMI_OP_P,
    NVSHMEMI_OP_PUT_SIGNAL,
    NVSHMEMI_OP_GET,
    NVSHMEMI_OP_G,
    NVSHMEMI_OP_FENCE,
    NVSHMEMI_OP_AMO,
    NVSHMEMI_OP_QUIET,
} nvshmemi_op_t;

typedef enum rdxn_ops {
    RDXN_OPS_AND = 0,
    RDXN_OPS_and = 0,
    RDXN_OPS_OR = 1,
    RDXN_OPS_or = 1,
    RDXN_OPS_XOR = 2,
    RDXN_OPS_xor = 2,
    RDXN_OPS_MIN = 3,
    RDXN_OPS_min = 3,
    RDXN_OPS_MAX = 4,
    RDXN_OPS_max = 4,
    RDXN_OPS_SUM = 5,
    RDXN_OPS_sum = 5,
    RDXN_OPS_PROD = 6,
    RDXN_OPS_prod = 6
} rdxn_ops_t;

typedef enum {
    NVSHMEMI_AMO_ACK = 1,
    NVSHMEMI_AMO_INC,
    NVSHMEMI_AMO_SET,
    NVSHMEMI_AMO_ADD,
    NVSHMEMI_AMO_AND,
    NVSHMEMI_AMO_OR,
    NVSHMEMI_AMO_XOR,
    NVSHMEMI_AMO_SIGNAL,
    NVSHMEMI_AMO_SIGNAL_SET,
    NVSHMEMI_AMO_SIGNAL_ADD,
    NVSHMEMI_AMO_END_OF_NONFETCH, //end of nonfetch atomics
    NVSHMEMI_AMO_FETCH,
    NVSHMEMI_AMO_FETCH_INC,
    NVSHMEMI_AMO_FETCH_ADD,
    NVSHMEMI_AMO_FETCH_AND,
    NVSHMEMI_AMO_FETCH_OR,
    NVSHMEMI_AMO_FETCH_XOR,
    NVSHMEMI_AMO_SWAP,
    NVSHMEMI_AMO_COMPARE_SWAP,
} nvshmemi_amo_t;

typedef enum { 
    NVSHMEMI_JOB_GPU_LDST_ATOMICS = 1,
    NVSHMEMI_JOB_GPU_LDST = 1 << 1,
    NVSHMEMI_JOB_GPU_PROXY = 1 << 2,
    NVSHMEMI_JOB_GPU_PROXY_CST = 1 << 3,
} nvshmemi_job_connectivity_t;

typedef enum {
    NVSHMEMI_PROXY_NONE = 0,
    NVSHMEMI_PROXY_MINIMAL = 1,
    NVSHMEMI_PROXY_FULL = 1 << 1,
} nvshmemi_proxy_status;
#endif
