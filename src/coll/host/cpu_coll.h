#ifndef NVSHMEMI_COLL_CPU_H
#define NVSHMEMI_COLL_CPU_H 1

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "util.h"
#include "nvshmem_api.h"
#include "nvshmemx_api.h"
#include "coll_shorthand.h"
#include "nvshmem_internal.h"

#include "rdxn.h"
#include "rdxn_on_stream.h"
#include "alltoall.h"
#include "alltoall_on_stream.h"
#include "barrier.h"
#include "barrier_on_stream.h"
#include "broadcast.h"
#include "broadcast_on_stream.h"
#include "collect.h"
#include "collect_on_stream.h"

/* macro definitions */
#define NVSHMEMI_COLL_CPU_STATUS_SUCCESS 0
#define NVSHMEMI_COLL_CPU_STATUS_ERROR 1

/* structs */
typedef struct nvshmemi_coll_cpu_support_modes {
    int none;
    int intra_node_intra_sock;
    int intra_node_inter_sock;
    int inter_node_inter_sock;
} nvshmemi_coll_support_cpu_modes_t;

typedef struct cpu_coll_cuda_ipc_exch {
    cudaIpcMemHandle_t mem_handle;
    cudaIpcEventHandle_t evnt_handle;
} cpu_coll_cuda_ipc_exch_t;

typedef struct cpu_coll_info {
    char fname[512];
    int my_pid;
    int *peer_pids;
    int shm_fd;
    volatile void *shm_addr;
    volatile int *cpu_bcast_int_sync_arr;
    volatile int *cpu_bcast_int_data_arr;
    volatile int *gpu_bcast_int_sync_arr;
    void *ipc_shm_addr;
    void *own_shm_addr;
    cudaIpcEventHandle_t *peer_shm_evnt_handles;
    cpu_coll_cuda_ipc_exch_t *peer_handles;
    cudaEvent_t *peer_cuda_events;
} cpu_coll_info_t;

/* global vars */
extern char *cu_err_string;
extern cpu_coll_info_t nvshm_cpu_coll_info;
extern int nvshm_enable_cpu_coll;
extern int nvshm_enable_p2p_cpu_coll;
extern int nvshm_use_p2p_cpu_push;
extern int nvshm_use_tg_for_stream_coll;
extern int nvshm_use_tg_for_cpu_coll;
extern int nvshm_cpu_coll_initialized;
extern int nvshm_cpu_coll_offset_reqd;
extern int nvshm_cpu_coll_sync_reqd;
extern int nvshm_cpu_rdxn_seg_size;
extern int cpu_shm_size;
extern int cpu_ipc_shm_size;
extern int nvshm_rdx_num_tpb;
extern int nvshm_use_p2p_cpu_rdxn_allgather;
extern int nvshm_use_p2p_cpu_rdxn_od_gather;

/* function declarations */
int nvshmemi_coll_common_cpu_read_env();
int nvshmemi_coll_common_cpu_init_memory();
int nvshmemi_coll_common_cpu_init();
int nvshmemi_coll_common_cpu_finalize();

int bcast_sync(int root, int val);

#define NVSHMEMI_COLL_CPU_ERR_POP()                                                        \
    do {                                                                                   \
        fprintf(stderr, "[pe = %d] Error at %s:%d in %s\n", nvshmem_state->mype, __FILE__, \
                __LINE__, __FUNCTION__);                                                   \
        fflush(stderr);                                                                    \
        goto fn_fail;                                                                      \
    } while (0)

#define NVSHMEMI_COLL_CPU_CUDA_ERR_POP(ERRNO)                                                   \
    do {                                                                                        \
        fprintf(stderr, "[pe = %d] Cuda error at %s:%d in %s\n", nvshmem_state->mype, __FILE__, \
                __LINE__, __FUNCTION__);                                                        \
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(ERRNO));                              \
        fflush(stderr);                                                                         \
        goto fn_fail;                                                                           \
    } while (0)

#define NVSHMEMI_COLL_CPU_CU_ERR_POP(ERRNO)                                                     \
    do {                                                                                        \
        fprintf(stderr, "[pe = %d] Cuda error at %s:%d in %s\n", nvshmem_state->mype, __FILE__, \
                __LINE__, __FUNCTION__);                                                        \
        cuGetErrorString(ERRNO, &cu_err_string);                                                \
        fprintf(stderr, "Error: %s\n", cu_err_string);                                          \
        fflush(stderr);                                                                         \
        goto fn_fail;                                                                           \
    } while (0)

#endif /* NVSHMEMI_COLL_CPU_H */
