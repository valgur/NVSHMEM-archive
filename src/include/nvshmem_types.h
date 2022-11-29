#ifndef NVSHMEM_TYPES_H
#define NVSHMEM_TYPES_H

typedef int32_t nvshmem_team_t;

typedef struct {
    int num_contexts;
} nvshmem_team_config_t;

#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"

template <rdxn_ops_t op>
inline ncclRedOp_t nvshmemi_get_nccl_op();

template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_SUM>() {
    return ncclSum;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_PROD>() {
    return ncclProd;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_MIN>() {
    return ncclMin;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_MAX>() {
    return ncclMax;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_AND>() {
    return ncclNumOps;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_OR>() {
    return ncclNumOps;
}
template <>
inline ncclRedOp_t nvshmemi_get_nccl_op<RDXN_OPS_XOR>() {
    return ncclNumOps;
}

/* Reduction datatypes */
/*
 * ncclChar is an unsigned type. char in c++ can be signed or unsigned
 * so pick the "right" nccl type depending on the implementation of char.
 */
template <typename T>
inline ncclDataType_t nvshmemi_get_nccl_dt();

template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<char>() {
#if (CHAR_MIN == 0)
    return ncclUint8;
#else
    return ncclChar;
#endif
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<signed char>() {
    return ncclChar;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<short>() {
    return ncclNumTypes;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<int>() {
    return ncclInt;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<long>() {
    return ncclInt64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<long long>() {
    return ncclInt64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned char>() {
    return ncclUint8;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned short>() {
    return ncclNumTypes;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned int>() {
    return ncclUint32;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned long>() {
    return ncclUint64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<unsigned long long>() {
    return ncclUint64;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<float>() {
    return ncclFloat;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<double>() {
    return ncclDouble;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<long double>() {
    return ncclNumTypes;
}
#ifdef NVSHMEM_COMPLEX_SUPPORT
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<complex double>() {
    return ncclNumTypes;
}
template <>
inline ncclDataType_t nvshmemi_get_nccl_dt<complex float>() {
    return ncclNumTypes;
}
#endif

struct nccl_function_table {
    ncclResult_t (*GetVersion)(int* version);
    const char* (*GetErrorString)(ncclResult_t result);
    ncclResult_t (*GetUniqueId)(ncclUniqueId* uniqueId);
    ncclResult_t (*CommInitRank)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
    ncclResult_t (*CommDestroy)(ncclComm_t comm);
    ncclResult_t (*AllReduce)(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                              cudaStream_t stream);
    ncclResult_t (*Broadcast)(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, int root, ncclComm_t comm,
                              cudaStream_t stream);
    ncclResult_t (*AllGather)(const void* sendbuff, void* recvbuff, size_t sendcount,
                              ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t (*GroupStart)();
    ncclResult_t (*GroupEnd)();
    ncclResult_t (*Send)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
                         ncclComm_t comm, cudaStream_t stream);
    ncclResult_t (*Recv)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                         ncclComm_t comm, cudaStream_t stream);
};

extern struct nccl_function_table nccl_ftable;

#endif /* NVSHMEM_USE_NCCL */

typedef struct {
    int step1_sendto;
    int* step1_recvfrom;
    int step1_nrecvs;
    int** step2_nbrs;
    int step2_nphases;
} nvshmemi_reduce_recexch_t;

typedef struct {
    int my_pe;
    int start, stride, size;
    int team_idx;
    nvshmem_team_config_t config;
    long config_mask;
#ifdef NVSHMEM_USE_NCCL
    ncclComm_t nccl_comm;
#endif
    nvshmemi_reduce_recexch_t reduce_recexch;
    size_t rdxn_count;
    uint32_t ll_flag;
    uint64_t alltoall_pwrk[2];
    uint64_t alltoall_count;
    uint64_t bcast_count;
    uint64_t fcollect_count;
    uint32_t fcollect_ll_flag;
    bool are_gpus_p2p_connected;
    /*size_t                       contexts_len;
    struct shmem_transport_ctx_t **contexts;*/
} nvshmemi_team_t;

typedef struct gpu_coll_env_params {
    int gpu_intm_rdxn_size;
    int reduce_recexch_kval;
} gpu_coll_env_params_t;

#endif /* NVSHMEM_TYPES_H */
