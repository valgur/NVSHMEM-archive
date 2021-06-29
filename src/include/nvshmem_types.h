#ifndef NVSHMEM_TYPES_H
#define NVSHMEM_TYPES_H

typedef int32_t nvshmem_team_t;

typedef struct {
    int num_contexts;
} nvshmem_team_config_t;

#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"
/* Reduction operation types */
#define NCCL_REDOP_sum ncclSum
#define NCCL_REDOP_prod ncclProd
#define NCCL_REDOP_min ncclMin
#define NCCL_REDOP_max ncclMax
#define NCCL_REDOP_and -1
#define NCCL_REDOP_or -1
#define NCCL_REDOP_xor -1

/* Reduction datatypes */
#define NCCL_DT_char ncclChar
#define NCCL_DT_schar -1
#define NCCL_DT_short -1
#define NCCL_DT_int ncclInt
#define NCCL_DT_long ncclInt64
#define NCCL_DT_longlong ncclInt64
#define NCCL_DT_ptrdiff ncclUint64
#define NCCL_DT_uchar ncclUint8
#define NCCL_DT_ushort -1
#define NCCL_DT_uint ncclUint32
#define NCCL_DT_ulong ncclUint64
#define NCCL_DT_ulonglong ncclUint64
#define NCCL_DT_int8 ncclInt8
#define NCCL_DT_int16 -1
#define NCCL_DT_int32 ncclInt
#define NCCL_DT_int64 ncclInt64
#define NCCL_DT_uint8 ncclUint8
#define NCCL_DT_uint16 -1
#define NCCL_DT_uint32 ncclUint32
#define NCCL_DT_uint64 ncclUint64
#define NCCL_DT_size ncclUint64
#define NCCL_DT_float ncclFloat
#define NCCL_DT_double ncclDouble
#define NCCL_DT_longdouble -1
#define NCCL_DT_complexd -1
#define NCCL_DT_complexf -1

#else /* NVSHMEM_USE_NCCL */

/* Reduction operation types */
#define NCCL_REDOP_sum -1
#define NCCL_REDOP_prod -1
#define NCCL_REDOP_min -1
#define NCCL_REDOP_max -1
#define NCCL_REDOP_and -1
#define NCCL_REDOP_or -1
#define NCCL_REDOP_xor -1

/* Reduction datatypes */
#define NCCL_DT_char -1
#define NCCL_DT_schar -1
#define NCCL_DT_short -1
#define NCCL_DT_int -1
#define NCCL_DT_long -1
#define NCCL_DT_longlong -1
#define NCCL_DT_ptrdiff -1
#define NCCL_DT_uchar -1
#define NCCL_DT_ushort -1
#define NCCL_DT_uint -1
#define NCCL_DT_ulong -1
#define NCCL_DT_ulonglong -1
#define NCCL_DT_int8 -1
#define NCCL_DT_int16 -1
#define NCCL_DT_int32 -1
#define NCCL_DT_int64 -1
#define NCCL_DT_uint8 -1
#define NCCL_DT_uint16 -1
#define NCCL_DT_uint32 -1
#define NCCL_DT_uint64 -1
#define NCCL_DT_size -1
#define NCCL_DT_float -1
#define NCCL_DT_double -1
#define NCCL_DT_longdouble -1
#define NCCL_DT_complexd -1
#define NCCL_DT_complexf -1

typedef int ncclRedOp_t;
typedef int ncclDataType_t;
typedef int ncclComm_t;
typedef int ncclResult_t;
typedef int ncclUniqueId;
#define ncclSuccess 0

#endif /* NVSHMEM_USE_NCCL */

typedef struct {
    int my_pe;
    int start, stride, size;
    int psync_idx;
    nvshmem_team_config_t config;
    long config_mask;
    ncclComm_t nccl_comm;
    /*size_t                       contexts_len;
    struct shmem_transport_ctx_t **contexts;*/
} nvshmemi_team_t;

typedef struct gpu_coll_env_params {
    int gpu_intm_rdxn_size;
    int reduce_recexch_kval;
} gpu_coll_env_params_t;

#endif /* NVSHMEM_TYPES_H */
