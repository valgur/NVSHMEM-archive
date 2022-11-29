/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _INTERNAL_H
#define _INTERNAL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudawrap.h"
#include "bootstrap_internal.h"
#include "transport.h"
#include "common.h"
#include "nvshmem_common.cuh"
#include <map>
#include <vector>
#include <pthread.h>
#include <atomic>
#include <tuple>
#include "nvshmemx_error.h"
#include "util.h"
using namespace std;

#define MAX_PEER_STREAMS 3

#define MAX_TRANSPORT_EP_COUNT 1

#define SYMMETRIC_SIZE_DEFAULT 1024 * 1024 * 1024

#define MAXPATHSIZE 1024
#define MAX_BUSID_SIZE 16

#define NUM_G_BUF_ELEMENTS 1024 * 1024
#define MAX_PES_PER_GPU 48

#define G_COALESCING_BUF_SIZE NUM_G_BUF_ELEMENTS *NVSHMEMI_WARP_SIZE * sizeof(uint64_t)

#define NVSHMEMI_CHECK_INIT_STATUS()                                                        \
    do {                                                                                    \
        if (nvshmemi_is_nvshmem_initialized == false)                                       \
            ERROR_EXIT("NVSHMEM API called before NVSHMEM initialization has completed\n"); \
    } while (0)

typedef struct {
    int multi_processor_count;
    int cooperative_launch;
} cuda_device_attributes_t;

typedef struct {
    cudaStream_t stream;
    cudaEvent_t begin_event;
    cudaEvent_t end_event;
} collective_launch_params_t;

typedef struct nvshmemi_mps_shmdata_t {
    volatile size_t nprocesses;
    volatile atomic<int> barrier;
    volatile atomic<bool> sense;
    volatile cudaIpcEventHandle_t event_handle[MAX_PES_PER_GPU];
} nvshmemi_mps_shmdata;

typedef struct nvshmemi_shared_memory_info_t {
    void *addr;
    size_t size;
    int shm_fd;
} nvshmemi_shared_memory_info;

typedef struct nvshmemi_state_dec {
    /*PE state*/
    int mype;
    int npes;
    int mype_node;
    int npes_node;
    /*environment info*/
    char *prefix;
    /*device state*/
    CUdevice cudevice;
    int device_id;
    CUcontext cucontext;
    /*symmetric heap state*/
    size_t heap_size;
    void *heap_base;
    void *global_heap_base; /* Used when using VMM API */
    /* registered local memory state */
    size_t registered_buffer_array_size;
    size_t registered_buffer_array_used;
    nvshmem_local_buf_handle_t **registered_buffers;
    pthread_rwlock_t registered_buffer_lock;
    bool host_memory_registration_supported;

    void **peer_heap_base_actual;
    void **peer_heap_base;
    void *heap_mspace;
    /* variables for VMM */
#if CUDA_VERSION >= 11000
    vector<CUmemGenericAllocationHandle> cumem_handles;
#endif
    size_t physical_heap_size;
    vector<vector<nvshmem_mem_handle> > handles;
    vector<tuple<size_t, void *, size_t> > idx_in_handles;
    /*transport info*/
    uint32_t atomic_host_endian_min_size;
    int transport_bitmap;
    int *transport_map;
    struct nvshmem_transport_pe_info *pe_info;
    struct nvshmem_transport **transports;
    int *selected_transport_for_rma;
    int *selected_transport_for_amo;
    /*consolidated rma ops*/
    rma_handle *rma;
    amo_handle *amo;
    fence_handle *fence;
    quiet_handle *quiet;
    /*scratch space*/
    char *scratch_space;
    size_t scratch_size;
    int *scratch;
    cuda_device_attributes_t cu_dev_attrib;
    collective_launch_params_t claunch_params;
    cudaStream_t my_stream;
    // proxy
    void *proxy;
    cudaStream_t *custreams;
    cudaEvent_t *cuevents;
    /* MPS support */
    cudaEvent_t mps_event;
    cudaEvent_t
        same_gpu_other_pe_mps_events[MAX_PES_PER_GPU - 1]; /* CUDA IPC mapped mps_events from the
                                                              PEs sharing the same GPU */
    nvshmemi_shared_memory_info shm_info;

    nvshmemi_state_dec()
        :
#if CUDA_VERSION >= 11000
          cumem_handles(),
#endif
          handles(),
          idx_in_handles() {
    }
    bool used_internal_streams;
} nvshmemi_state_t;

typedef struct {
    volatile uint64_t data;
    volatile uint64_t flag;
} g_elem_t;

typedef struct {
    int error_checks;
} nvshmem_options_t;

typedef int (*nvshmemi_state_change_handler_fn_t)(void);

extern nvshmemi_state_t *nvshmemi_state;
extern bootstrap_handle_t nvshmemi_boot_handle;
extern int nvshmemi_init_counter;
extern nvshmem_options_t nvshmem_options;
extern int nvshmemi_job_connectivity;
extern int nvshmemi_cuda_driver_version;
extern int nvshmemi_use_nccl;
extern bool nvshmemi_is_mps_available;
extern bool nvshmemi_use_cuda_vmm;
extern int nccl_version;
extern long nvshmemi_max_teams;
extern size_t cumem_granularity;
extern size_t log2_cumem_granularity;

#include "nvshmemi_team.h"

int nvshmemi_proxy_level(nvshmemi_state_t *state);
int nvshmemi_common_init(nvshmemi_state_t *state);
int nvshmemi_init_g_buffer();
int nvshmemi_init_device_state(nvshmemi_state_t *state);
int nvshmemi_set_device_state(nvshmemi_device_state_t *);
int nvshmemi_setup_local_heap(nvshmemi_state_t *state);
int nvshmemi_setup_symmetric_heap(nvshmemi_state_t *state);
int nvshmemi_setup_connections(nvshmemi_state_t *state);
int nvshmemi_cleanup_symmetric_heap(nvshmemi_state_t *state);
int nvshmemi_setup_collective_launch(nvshmemi_state_t *state);
int nvshmemi_teardown_collective_launch(nvshmemi_state_t *state);
int nvshmemi_setup_mops_kernels(nvshmemi_state_t *state);
extern "C" {
void *nvshmemi_malloc(size_t size);
bool nvshmemi_is_version_compatible(const nvshmemi_version_t version_1,
                                    const nvshmemi_version_t version_2);
}
void *nvshmemi_calloc(size_t count, size_t size);
void *nvshmemi_align(size_t alignment, size_t size);
void nvshmemi_free(void *ptr);
void nvshmemi_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                  cudaStream_t cstrm);
extern "C" {
__device__ void nvshmemi_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
}

void nvshmemi_barrier_all();

int nvshmemi_proxy_init(nvshmemi_state_t *state, int proxy_level);
int nvshmemi_proxy_finalize(nvshmemi_state_t *state);

struct nvshmem_mem_handle *nvshmemi_get_registered_buffer_handle(void *addr, size_t *len);

static inline void nvshmemi_get_local_mem_handle(nvshmem_mem_handle_t **handle, size_t *len,
                                                 void *addr, int transport_idx) {
    nvshmem_mem_handle_t *handle_ptr;
    size_t max_len = SIZE_MAX;

    if (addr >= nvshmemi_state->heap_base &&
        (addr < (void *)((char *)nvshmemi_state->heap_base + nvshmemi_state->heap_size))) {
        /* heap lookup code. */
        size_t offset = (char *)addr - (char *)nvshmemi_state->heap_base;
        size_t addr_idx = offset >> log2_cumem_granularity;
        size_t handle_idx = std::get<0>(nvshmemi_state->idx_in_handles[addr_idx]);
        void *handle_start_addr = std::get<1>(nvshmemi_state->idx_in_handles[addr_idx]);
        size_t handle_size = std::get<2>(nvshmemi_state->idx_in_handles[addr_idx]);
        *handle =
            &nvshmemi_state->handles[handle_idx][nvshmemi_state->mype * NVSHMEM_TRANSPORT_COUNT +
                                                 transport_idx];
        if (len) *len = handle_size - ((char *)addr - (char *)handle_start_addr);
    } else {
        /* registered buffer lookup code */
        handle_ptr = nvshmemi_get_registered_buffer_handle(addr, len);
        if (handle_ptr) {
            *handle = handle_ptr;
        } else {
            *handle = NULL;
        }
    }

    if (transport_idx == NVSHMEM_TRANSPORT_ID_IBRC || transport_idx == NVSHMEM_TRANSPORT_ID_IBDEVX
#ifdef NVSHMEM_IBGDA_SUPPORT
        || transport_idx == NVSHMEM_TRANSPORT_ID_GIC
#endif
    ) {
        /* 1 GB Max*/
        max_len = 1ULL << 30;
    }

    if (len) *len = *len < max_len ? *len : max_len;
}

static inline void nvshmemi_get_remote_mem_handle(nvshmem_mem_handle_t **handle, size_t *len,
                                                  void *addr, int pe, int transport_idx) {
    size_t offset = (char *)addr - (char *)nvshmemi_state->heap_base;
    size_t addr_idx = offset >> log2_cumem_granularity;
    size_t handle_idx = std::get<0>(nvshmemi_state->idx_in_handles[addr_idx]);
    void *handle_start_addr = std::get<1>(nvshmemi_state->idx_in_handles[addr_idx]);
    size_t handle_size = std::get<2>(nvshmemi_state->idx_in_handles[addr_idx]);

    *handle = &nvshmemi_state->handles[handle_idx][pe * NVSHMEM_TRANSPORT_COUNT + transport_idx];
    if (len) *len = handle_size - ((char *)addr - (char *)handle_start_addr);
}
/* rptr is symmetric address on the local pe
   lptr is local address - either symmetric or not */
static inline void nvshmemi_process_multisend_rma(struct nvshmem_transport *tcurr, int transport_id,
                                                  int pe, rma_verb_t verb, void *rptr, void *lptr,
                                                  size_t size, bool is_proxy) {
    rma_memdesc_t localdesc, remotedesc;
    rma_bytesdesc_t bytes;
    bytes.srcstride = 1;
    bytes.deststride = 1;
    bytes.elembytes = 1;
    size_t local_chunk_size, remote_chunk_size, size_remaining;
    size_t chunk_size;
    size_remaining = size;
    int status;

    while (size_remaining) {
        localdesc.ptr = lptr;
        NVSHMEMU_UNMAPPED_PTR_TRANSLATE(remotedesc.ptr, rptr, pe);
        remotedesc.offset = (char *)rptr - (char *)nvshmemi_state->heap_base;
        local_chunk_size = size_remaining;
        remote_chunk_size = size_remaining;
        nvshmemi_get_local_mem_handle(&localdesc.handle, &local_chunk_size, lptr, transport_id);
        nvshmemi_get_remote_mem_handle(&remotedesc.handle, &remote_chunk_size, rptr, pe,
                                       transport_id);
        chunk_size = min(local_chunk_size, min(remote_chunk_size, size_remaining));
        bytes.nelems = chunk_size;

        status = tcurr->host_ops.rma(tcurr, pe, verb, &remotedesc, &localdesc, bytes, is_proxy);
        if (unlikely(status)) {
            ERROR_PRINT("aborting due to error in process_channel_dma\n");
            exit(-1);
        }
        size_remaining -= chunk_size;
        lptr = (char *)lptr + chunk_size;
        rptr = (char *)rptr + chunk_size;
    }
}

extern "C" {
void nvshmemi_register_state_change_handler(nvshmemi_state_change_handler_fn_t fn);
}
int nvshmemi_update_device_state();

#endif
