#ifndef NVSHMEM_TYPES_H
#define NVSHMEM_TYPES_H

#if not defined __CUDACC_RTC__
#include <stdint.h>
#else
#include <cuda/std/cstdint>
#endif

typedef int32_t nvshmem_team_t;
typedef nvshmem_team_t nvshmemx_team_t;

#define INIT_HANDLE_BYTES 128
typedef struct {
    char content[INIT_HANDLE_BYTES];
} nvshmemx_init_handle_t;

typedef struct {
    size_t heap_size;
    int num_threads;
    int n_pes;
    int my_pe;
    void *mpi_comm;
    nvshmemx_init_handle_t handle;
} nvshmemx_init_attr_t;

typedef enum {
    NVSHMEMI_PE_DIST_ROUNDROBIN = 0,
    NVSHMEMI_PE_DIST_BLOCK,
    NVSHMEMI_PE_DIST_MISC
} nvshmemi_pe_dist_t;

typedef struct {
    int step1_sendto;
    int *step1_recvfrom;
    int step1_nrecvs;
    int **step2_nbrs;
    int step2_nphases;
} nvshmemi_reduce_recexch_t;

typedef struct {
    int num_contexts;
} nvshmem_team_config_t;

typedef struct {
    int my_pe;
    int start, stride, size;
    int team_idx;
    nvshmem_team_config_t config;
    long config_mask;
    void *nccl_comm; /* To be cast to ncclComm_t whenever used */
    nvshmemi_reduce_recexch_t reduce_recexch;
    size_t rdxn_count;
    uint32_t ll_flag;
    uint64_t alltoall_pwrk[2];
    uint64_t alltoall_count;
    uint64_t bcast_count;
    uint64_t bcast_sync_offset;
    uint64_t fcollect_count;
    uint32_t fcollect_ll_flag;
    bool are_gpus_p2p_connected;
    bool is_team_node;
    nvshmem_team_t team_node;
    bool is_team_same_mype_node;
    nvshmem_team_t team_same_mype_node;
    nvshmemi_pe_dist_t pe_dist;
    /*size_t                       contexts_len;
    struct shmem_transport_ctx_t **contexts;*/
} nvshmemi_team_t;

typedef struct gpu_coll_env_params {
    int gpu_intm_rdxn_size;
    int reduce_recexch_kval;
    int bcast_tree_kval;
    int bcast_algo;
    int reduce_algo;
} gpu_coll_env_params_t;

typedef struct {
    uint64_t signal;
    uint64_t caller;
    uint64_t signal_addr;
    uint64_t signal_val_found;
    uint64_t signal_val_expected;
} nvshmemi_timeout_t;

typedef struct {
    int mype;
    int npes;
    int node_mype;
    int node_npes;
    nvshmemi_pe_dist_t pe_dist;
    int *p2p_attrib_native_atomic_support;
    int proxy;
    int atomics_sync;
    int job_connectivity;
    bool proxy_ops_are_ordered;
    bool atomics_complete_on_quiet;
    void *heap_base;
    size_t heap_size;
    void **peer_heap_base;
    void **peer_heap_base_actual;
    bool symmetric_heap_kind;
    bool enable_rail_opt;
    uint32_t atomics_le_min_size;

    nvshmemi_timeout_t *timeout;
    unsigned long long *test_wait_any_start_idx_ptr;

    nvshmemi_team_t **team_pool;
    long *psync_pool;
    long *sync_counter;

    int barrier_dissem_kval;
    int barrier_tg_dissem_kval;
    size_t bcast_ll_threshold;
    size_t fcollect_ll_threshold;
    gpu_coll_env_params_t gpu_coll_env_params_var;

    /* channel */
    void *proxy_channels_buf; /* requests are written in this buffer */
    char *proxy_channel_g_buf;
    char *proxy_channel_g_coalescing_buf;
    uint64_t *proxy_channel_g_buf_head_ptr; /* next location to be assigned to a thread */
    uint64_t proxy_channel_g_buf_size;      /* Total size of g_buf in bytes */
    uint64_t proxy_channel_g_buf_log_size;  /* Total size of g_buf in bytes */
    uint64_t *proxy_channels_issue;         /* last byte of the last request */
    uint64_t *
        proxy_channels_complete; /* shared betwen CPU and GPU threads - only write by CPU thread and
                                      read by GPU threads. This is allocated on the system memory */
    uint64_t *proxy_channels_complete_local_ptr; /* shared only between GPU threads */
    uint64_t *proxy_channels_quiet_issue;
    uint64_t *proxy_channels_quiet_ack;
    uint64_t *proxy_channels_cst_issue;
    uint64_t *proxy_channels_cst_ack;
    uint64_t proxy_channel_buf_size; /* Maximum number of inflight requests in bytes OR
                                                   maximum channel length */
    uint32_t proxy_channel_buf_logsize;
    int *global_exit_request_state;
    int *global_exit_code;

    bool ibgda_is_initialized;
} nvshmemi_device_state_t;

#endif /* NVSHMEM_TYPES_H */
